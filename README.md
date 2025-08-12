# alt-oracle

- parse mbox and store emails in vector box - I might have to use intemediatery storage
- create embeddings of the vector box
- test a few seches using index
- write a poc program to send the the chunks to llm and ask for summary

tools:
    filter_emails_using_pandas
    filter_emails_using_rag
    summarize_answer

nodes:
    MailSniff
        llm_node
        tools_node

state:
