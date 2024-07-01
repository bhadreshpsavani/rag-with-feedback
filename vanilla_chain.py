from datetime import datetime
from langchain import LLMChain

# from langchain.chat_models import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory


def get_llm_chain(system_prompt: str, memory: ConversationBufferMemory) -> LLMChain:
    """Return a basic LLMChain with memory."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "\nIt's currently {time}.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    ).partial(time=lambda: str(datetime.now()))
    # llm = ChatOpenAI(temperature=0.7)
    llm = ChatBedrock(
        model_id="mistral.mistral-7b-instruct-v0:2",
        model_kwargs={"temperature": 0.1},
        region_name="us-east-1"
    )
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    return chain


if __name__ == "__main__":
    chain, _ = get_llm_chain()
    print(chain.invoke({"input": "Hi there, I'm a human!"})["text"])
    print(chain.invoke({"input": "What's your name?"})["text"])
