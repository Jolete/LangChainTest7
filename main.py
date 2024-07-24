from modules.environment.environment_utilities import (
    load_environment_variables,
    verify_environment_variables,
)
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import RetrievalQA


# Main program
try:
   
    #region Load environtment

    # Load environment variables using the utility
    env_vars = load_environment_variables()
    
    # Verify the environment variables
    if not verify_environment_variables(env_vars):
        raise ValueError("Some environment variables are missing!")

    llm  = OpenAI(
        openai_api_key=env_vars["OPEN_AI_SECRET_KEY"])

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


    #region create documents and new vector
    # print("\nAra creem un nou vector i els documents:\n")
    # documents = [
    #     Document(
    #         page_content="Text to be indexed",
    #         metadata={"source": "local"}
    #     )
    # ]    

    # new_vector = Neo4jVector.from_documents(
    #     documents,
    #     embedding_provider,
    #     graph=graph,
    #     index_name="myVectorIndex",
    #     node_label="Chunk",
    #     text_node_property="text",
    #     embedding_node_property="embedding",
    #     create_id_index=True,
    # )

    #endregion

    #region create retrieval
    plot_retriever = RetrievalQA.from_llm(
        llm=llm,
        retriever=movie_plot_vector.as_retriever(),
        verbose=True,
        return_source_documents=True
    )

    response = plot_retriever.invoke(
        {"query": "A movie where a mission to the moon goes wrong"}
    )

    print("\n")
    print(response)
    #endregion

except Exception as e:
    print(f"An unexpected error occurred: {e}")