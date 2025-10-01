import json
from enum import Enum

from agents import function_tool
from agents import RunContextWrapper

from onyx.agents.agent_search.dr.models import GeneratedImage
from onyx.chat.turn.models import MyContext
from onyx.server.query_and_chat.streaming_models import ImageGenerationToolDelta
from onyx.server.query_and_chat.streaming_models import ImageGenerationToolHeartbeat
from onyx.server.query_and_chat.streaming_models import ImageGenerationToolStart
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import SectionEnd
from onyx.utils.logger import setup_logger

logger = setup_logger()


class ImageShape(str, Enum):
    SQUARE = "square"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


@function_tool
def image_generation_tool(
    run_context: RunContextWrapper[MyContext], prompt: str, shape: str = "square"
) -> str:
    """
    Generate an image from a text prompt using AI image generation models.

    Args:
        prompt: The text description of the image to generate
        shape: The desired image shape - 'square', 'portrait', or 'landscape'
    """
    # Get the image generation tool from context
    image_generation_tool = run_context.context.run_dependencies.image_generation_tool
    if image_generation_tool is None:
        raise RuntimeError("Image generation tool not available in context")

    index = run_context.context.current_run_step + 1
    emitter = run_context.context.run_dependencies.emitter

    # Emit start event
    emitter.emit(
        Packet(
            ind=index,
            obj=ImageGenerationToolStart(type="image_generation_tool_start"),
        )
    )

    # Prepare tool arguments
    tool_args = {"prompt": prompt}
    if shape != "square":  # Only include shape if it's not the default
        tool_args["shape"] = shape

    # Run the actual image generation tool with heartbeat handling
    generated_images = []
    heartbeat_count = 0

    for tool_response in image_generation_tool.run(**tool_args):
        # Handle heartbeat responses
        if tool_response.id == "image_generation_heartbeat":
            # Emit heartbeat event for every iteration
            emitter.emit(
                Packet(
                    ind=index,
                    obj=ImageGenerationToolHeartbeat(
                        type="image_generation_tool_heartbeat"
                    ),
                )
            )
            heartbeat_count += 1
            logger.debug(f"Image generation heartbeat #{heartbeat_count}")
            continue

        # Process the tool response to get the generated images
        if tool_response.id == "image_generation_response":
            image_generation_responses = tool_response.response
            generated_images = [
                GeneratedImage(
                    revised_prompt=img.revised_prompt,
                    url=img.url,
                    image_data=img.image_data,
                )
                for img in image_generation_responses
            ]
            break

    if not generated_images:
        raise RuntimeError("No images were generated")

    # Emit final result
    emitter.emit(
        Packet(
            ind=index,
            obj=ImageGenerationToolDelta(
                type="image_generation_tool_delta", images=generated_images
            ),
        )
    )

    # Emit section end
    emitter.emit(
        Packet(
            ind=index,
            obj=SectionEnd(
                type="section_end",
            ),
        )
    )

    run_context.context.current_run_step = index + 1

    # Return the first generated image data
    first_image = generated_images[0]
    return json.dumps(
        {
            "revised_prompt": first_image.revised_prompt,
            "url": first_image.url,
            "image_data": first_image.image_data,
        }
    )
