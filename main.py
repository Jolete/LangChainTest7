from modules.environment.environment_utilities import (
    load_environment_variables,
    verify_environment_variables,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector



# Main program
try:
   
    #region Load environtment

    # Load environment variables using the utility
    env_vars = load_environment_variables()
    
    # Verify the environment variables
    if not verify_environment_variables(env_vars):
        raise ValueError("Some environment variables are missing!")

    embedding_provider = OpenAIEmbeddings(
        openai_api_key=env_vars["OPEN_AI_SECRET_KEY"])
  
    #endregion

    #region neo4j db 
    
    graph = Neo4jGraph(
        url=env_vars["NEO4J_URI"],
        username=env_vars["NEO4J_USERNAME"],
        password=env_vars["NEO4J_PASSWORD"]
    )

    #endregion

    #region create vector 
    movie_plot_vector = Neo4jVector.from_existing_index(
        embedding_provider ,
        graph=graph,
        index_name="moviePlots",
        embedding_node_property="plotEmbedding",
        text_node_property="plot",
    )
    #endregion

    query = "A movie where aliens land and attack earth."
    result = movie_plot_vector.similarity_search(query, 1)
    for doc in result:
        print(doc.metadata["title"], "-", doc.page_content)

        
    # #region Session & Prompt & Memory & Tools
    # SESSION_ID = str(uuid4())
    # print("\nSession ID: {SESSION_ID}\n")

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are a movie expert. You find movies from a genre or plot.",
    #         ),
    #         ("human", "{input}"),
    #     ]
    # )

    # movie_chat = prompt | llm | StrOutputParser()

    # youtube = YouTubeSearchTool()

    # def get_memory(session_id):
    #     return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

    # def call_trailer_search(input):
    #     input = input.replace(",", " ")
    #     return youtube.run(input)

    # tools = [
    #     Tool.from_function(
    #         name="Movie Chat",
    #         description="For when you need to chat about movies. The question will be a string. Return a string.",
    #         func=movie_chat.invoke,
    #     ),
    #     Tool.from_function(
    #         name="Movie Trailer Search",
    #         description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
    #         func=call_trailer_search,
    #     ),
    # ]
    # #endregion

    # #region Agent configuration

    # agent_prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(llm, tools, agent_prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools)

    # chat_agent = RunnableWithMessageHistory(
    #     agent_executor,
    #     get_memory,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    # )
    # #endregion 

    # while True:
    #     q = input("> ")

    #     response = chat_agent.invoke(
    #         {
    #             "input": q
    #         },
    #         {"configurable": {"session_id": SESSION_ID}},
    #     )
        
    #     print(response["output"])

except Exception as e:
    print(f"An unexpected error occurred: {e}")