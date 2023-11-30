from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import json
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
  title="Sura señales",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)
 

# template = """Summarize the following question based on the context:

# Question : {question}

# Context: {context}"""

# prompt = ChatPromptTemplate.from_template(template)

RESULTS_PER_QUESTION = 2

ddg_search = DuckDuckGoSearchAPIWrapper()

@app.post("/resaearch-assistant")
async def research_assistant(prompt_input: str):

    def web_search(query:str, num_results: int = RESULTS_PER_QUESTION):
        results = ddg_search.results(query, num_results)
        return [r["link"] for r in results]

    SUMMARY_TEMPLATE = """{text} 
    -----------
    Using the above text, answer in short the following question: 
    > {question}
    -----------
    If the question cannot be answered using the text, please summarize the text. Include all factual information, numbers, statistics, socio-environmental, economic, political impacts, etc., if available."""

    SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


    def scrape_text(url: str):
        # Send a GET request to the webpage
        try:
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the content of the request with BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract all text from the webpage
                page_text = soup.get_text(separator=" ", strip=True)

                # Print the extracted text
                return page_text
            else:
                return f"Failed to retrieve the webpage: Status code {response.status_code}"
        except Exception as e:
            print(e)
            return f"Failed to retrieve the webpage: {e}"


    scrape_and_summarize_chain = RunnablePassthrough.assign(
        summary = RunnablePassthrough.assign(
        text=lambda x: scrape_text(x["url"])[:10000]
    ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
    ) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

    web_search_chain = RunnablePassthrough.assign(
        urls = lambda x: web_search(x['question'])
    ) | (lambda x: [{'question': x['question'], 'url': u} for u in x['urls']]) | scrape_and_summarize_chain.map() 

    ''''''

    SEARCH_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                'user',
                'Write 3 google search queries to search online information or recent news '
                'about the following: {question}\n'
                'You must respond with a list of strings in the following format: '
                '["query1", "query2", "query3"].',
            ),
        ]
    )

    search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() |json.loads

    full_research_chain = search_question_chain | (lambda x: [{'question': q} for q in x]) | web_search_chain.map()


    # full_research_chain.invoke(
    #     {
    #         "question" : "Qué tipo de recursos naturales están escasos en Latino America?"
    #     }
    # )

    WRITER_SYSTEM_PROMPT = "You are a research assistant and your sole purpose is to search for recent reports or news on a specific topic and that they are from reliable sources."  


    RESEARCH_REPORT_TEMPLATE = """Information:
    --------
    {research_summary}
    --------
    Using the information above, answer the following question or topic: "{question}" in a detailed report perform an analysis of each topic trending its impact over the next 3 years -- \
    The report should focus on the answer to the question, should be well structured, informative, \
    in depth, with facts and numbers if available and a minimum of 300 words.
    You should strive to write the report as long as you can using all relevant and necessary information provided.
    You must write the report with markdown syntax.
    You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
    Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
    Please do your best, this is very important for the company since future decisions will be made from this information."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("user", RESEARCH_REPORT_TEMPLATE)
        ]
    )

    def collapse_list_of_lists(list_of_lists):
        content = []
        for l in list_of_lists:
            content.append("\n\n".join(l))
        
        return "\n\n".join(content)

    chain = RunnablePassthrough.assign(
        research_summary = full_research_chain | collapse_list_of_lists
    ) | prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()

    response_urls_and_sumary = full_research_chain.invoke(
        {
            "question": prompt_input
        }
    )

    response_trends = chain.invoke(
        {
            "question": prompt
        }
    )


    return {
            "urls_and_sumary": response_urls_and_sumary, 
            "trends": response_trends
        }







