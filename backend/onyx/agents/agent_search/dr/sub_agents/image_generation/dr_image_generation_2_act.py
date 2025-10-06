import json
from collections.abc import Iterable
from datetime import datetime
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from onyx.agents.agent_search.dr.models import GeneratedImage
from onyx.agents.agent_search.dr.models import IterationAnswer
from onyx.agents.agent_search.dr.sub_agents.states import BranchInput
from onyx.agents.agent_search.dr.sub_agents.states import BranchUpdate
from onyx.agents.agent_search.models import GraphConfig
from onyx.agents.agent_search.shared_graph_utils.utils import (
    get_langgraph_node_log_string,
)
from onyx.agents.agent_search.shared_graph_utils.utils import write_custom_event
from onyx.file_store.utils import build_frontend_file_url
from onyx.file_store.utils import save_files
from onyx.server.query_and_chat.streaming_models import ImageGenerationToolHeartbeat
from onyx.tools.tool_implementations.images.image_generation_tool import (
    IMAGE_GENERATION_HEARTBEAT_ID,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    IMAGE_GENERATION_RESPONSE_ID,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationResponse,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationTool,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageShape,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()


_VALID_IMAGE_SHAPES = {shape.value for shape in ImageShape}
_SHAPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    ImageShape.LANDSCAPE.value: ("landscape", "horizontal", "wide", "widescreen"),
    ImageShape.PORTRAIT.value: ("portrait", "vertical", "tall"),
    ImageShape.SQUARE.value: ("square",),
}


def _find_keyword_shape(texts: Iterable[str]) -> str | None:
    for text in texts:
        normalized = text.lower()
        for shape_value, keywords in _SHAPE_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                logger.debug(
                    "Detected image shape '%s' from text snippet '%s'",
                    shape_value,
                    text[:80],
                )
                return shape_value
    return None


def _normalize_shape(raw_shape: str | None) -> str | None:
    if not raw_shape:
        return None
    candidate = raw_shape.strip().lower()
    if candidate in _VALID_IMAGE_SHAPES:
        return candidate
    logger.debug(f"Unsupported image shape requested: {raw_shape}")
    return None


def _extract_prompt_and_shape(
    raw_query: str,
    metadata: dict[str, str] | None,
    fallback_texts: Iterable[str] | None = None,
) -> tuple[str, str | None]:
    prompt = raw_query.strip()
    shape: str | None = None

    try:
        parsed_query = json.loads(raw_query)
    except json.JSONDecodeError:
        parsed_query = None

    if isinstance(parsed_query, dict):
        prompt_value = parsed_query.get("prompt")
        if isinstance(prompt_value, str) and prompt_value.strip():
            prompt = prompt_value

        shape_value = parsed_query.get("shape")
        if isinstance(shape_value, str):
            shape = _normalize_shape(shape_value)

    if not shape and metadata:
        metadata_shape = metadata.get("shape")
        shape = _normalize_shape(metadata_shape)

    if not shape:
        fallback_candidates = list(fallback_texts or []) + [prompt]
        shape = _normalize_shape(_find_keyword_shape(fallback_candidates))

    if not prompt:
        prompt = raw_query

    return prompt, shape


def _expected_dimensions(
    shape: ImageShape | None, model: str
) -> tuple[int | None, int | None]:
    if shape is None:
        return None, None

    if shape == ImageShape.LANDSCAPE:
        size = "1536x1024" if model == "gpt-image-1" else "1792x1024"
    elif shape == ImageShape.PORTRAIT:
        size = "1024x1536" if model == "gpt-image-1" else "1024x1792"
    else:
        size = "1024x1024"

    try:
        width_str, height_str = size.split("x")
        return int(width_str), int(height_str)
    except ValueError:
        logger.debug("Unable to parse expected size '%s'", size)
        return None, None


def image_generation(
    state: BranchInput,
    config: RunnableConfig,
    writer: StreamWriter = lambda _: None,
) -> BranchUpdate:
    """
    LangGraph node to perform a standard search as part of the DR process.
    """

    node_start_time = datetime.now()
    iteration_nr = state.iteration_nr
    parallelization_nr = state.parallelization_nr
    state.assistant_system_prompt
    state.assistant_task_prompt

    branch_query = state.branch_question
    if not branch_query:
        raise ValueError("branch_query is not set")

    graph_config = cast(GraphConfig, config["metadata"]["config"])
    graph_config.inputs.prompt_builder.raw_user_query
    graph_config.behavior.research_type

    if not state.available_tools:
        raise ValueError("available_tools is not set")

    image_tool_info = state.available_tools[state.tools_used[-1]]
    image_tool = cast(ImageGenerationTool, image_tool_info.tool_object)

    logger.debug(
        f"Image generation start for {iteration_nr}.{parallelization_nr} at {datetime.now()}"
    )

    # Generate images using the image generation tool
    image_generation_responses: list[ImageGenerationResponse] = []

    fallback_texts = []
    raw_user_query = graph_config.inputs.prompt_builder.raw_user_query
    if isinstance(raw_user_query, str):
        fallback_texts.append(raw_user_query)

    image_prompt, image_shape = _extract_prompt_and_shape(
        branch_query,
        image_tool_info.metadata,
        fallback_texts,
    )

    image_shape_enum: ImageShape | None = None
    if image_shape:
        try:
            image_shape_enum = ImageShape(image_shape)
        except ValueError:
            logger.debug("Encountered invalid image shape value: %s", image_shape)
            image_shape_enum = None

    expected_width, expected_height = _expected_dimensions(
        image_shape_enum, image_tool.model
    )

    shape_for_text = image_shape_enum.value if image_shape_enum else None

    run_kwargs: dict[str, str] = {"prompt": image_prompt}
    if image_shape_enum:
        run_kwargs["shape"] = image_shape_enum.value

    for tool_response in image_tool.run(**run_kwargs):
        if tool_response.id == IMAGE_GENERATION_HEARTBEAT_ID:
            # Stream heartbeat to frontend
            write_custom_event(
                state.current_step_nr,
                ImageGenerationToolHeartbeat(
                    shape=image_shape_enum.value if image_shape_enum else None,
                    width=expected_width,
                    height=expected_height,
                ),
                writer,
            )
        elif tool_response.id == IMAGE_GENERATION_RESPONSE_ID:
            response = cast(list[ImageGenerationResponse], tool_response.response)
            image_generation_responses = response
            break

    # save images to file store
    file_ids = save_files(
        urls=[img.url for img in image_generation_responses if img.url],
        base64_files=[
            img.image_data for img in image_generation_responses if img.image_data
        ],
    )

    final_generated_images: list[GeneratedImage] = []
    for file_id, img in zip(file_ids, image_generation_responses):
        response_shape = img.shape
        if isinstance(response_shape, ImageShape):
            response_shape_value = response_shape.value
        else:
            response_shape_value = response_shape

        if not response_shape_value and image_shape_enum:
            response_shape_value = image_shape_enum.value

        width = img.width or expected_width
        height = img.height or expected_height

        final_generated_images.append(
            GeneratedImage(
                file_id=file_id,
                url=build_frontend_file_url(file_id),
                revised_prompt=img.revised_prompt,
                width=width,
                height=height,
                shape=response_shape_value,
            )
        )

    logger.debug(
        f"Image generation complete for {iteration_nr}.{parallelization_nr} at {datetime.now()}"
    )

    if not shape_for_text and final_generated_images:
        shape_for_text = final_generated_images[0].shape

    # Create answer string describing the generated images
    if final_generated_images:
        image_descriptions = []
        for i, img in enumerate(final_generated_images, 1):
            image_descriptions.append(f"Image {i}: {img.revised_prompt}")

        request_details = image_prompt
        if shape_for_text:
            request_details = f"{image_prompt} (shape: {shape_for_text})"

        answer_string = (
            f"Generated {len(final_generated_images)} image(s) based on the request: {request_details}\n\n"
            + "\n".join(image_descriptions)
        )
        reasoning = (
            f"Used image generation tool to create {len(final_generated_images)} image(s)"
            f" based on the user's request{' with shape ' + shape_for_text if shape_for_text else ''}."
        )
    else:
        answer_string = f"Failed to generate images for request: {image_prompt}"
        reasoning = "Image generation tool did not return any results."

    return BranchUpdate(
        branch_iteration_responses=[
            IterationAnswer(
                tool=image_tool_info.llm_path,
                tool_id=image_tool_info.tool_id,
                iteration_nr=iteration_nr,
                parallelization_nr=parallelization_nr,
                question=image_prompt,
                answer=answer_string,
                claims=[],
                cited_documents={},
                reasoning=reasoning,
                generated_images=final_generated_images,
            )
        ],
        log_messages=[
            get_langgraph_node_log_string(
                graph_component="image_generation",
                node_name="generating",
                node_start_time=node_start_time,
            )
        ],
    )
