from agents import function_tool
from agents.run_context import RunContextWrapper
from langchain.schema import SystemMessage
from langchain_core.messages import HumanMessage

from onyx.chat.turn.models import ChatTurnContext


@function_tool
def reasoning_tool(dependencies: RunContextWrapper[ChatTurnContext], query: str) -> str:
    """Use this tool as a private scratchpad to reason about the query and use the result
    generated to help you answer the query. Use this tool for complex reasoning and
    calculations.
    """
    answer = ""
    tokens = dependencies.context.run_dependencies.llm.stream(
        [
            SystemMessage(
                content=f"""
                You are a private scratchpad to reason about the query and use the result generated to help you answer the query.
                Use this tool for complex reasoning and calculations. The following is your question.
                Please reason about the query and use the result generated to help you answer the query.
                Use this tool for complex reasoning and calculations.
                {query}
                """
            ),
            HumanMessage(content=query),
        ]
    )
    for token in tokens:
        print(token.content)
        content = token.content
        if isinstance(content, str):
            answer += content
        elif isinstance(content, list):
            answer += "".join(str(item) for item in content)
    return answer
