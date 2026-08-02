# # Step 1: Importing library and API key

import os
from openai import OpenAI
from dotenv import load_dotenv
from IPython.display import Markdown, display, Javascript
from pprint import pprint, pformat
import gradio as gr
import requests
import json
from litellm import completion
import random
import chromadb
import uuid

#load environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PUSHOVER_USER=os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN=os.getenv("PUSHOVER_TOKEN")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing")
else:
    print(OPENAI_API_KEY[:8])
# create a client object
client = OpenAI()


# Step 2: Documentation - RAG and Vector DB set up
# Step 2a: Documentation acceptance


system_message = """
You are the AI Digital Twin of Karthik Swaminathan, MBA, acting as a personal assistant and context provider. 

CRITICAL OPERATIONAL CONSTRAINTS:
1. CLOSED-CONTEXT ONLY: Answer user queries relying EXCLUSIVELY on the verbatim text enclosed between the *** marks below.
2. NO EXTERNAL KNOWLEDGE OR ACRONYMS: Do not use your pre-trained internal knowledge to expand acronyms, companies, or concepts (e.g., UHG, UnitedHealth, CVS, etc.) that are not explicitly written inside the text below.
3. ANTI-HALLUCINATION GUARDRAIL: If a user query contains ANY terms, acronyms, or topics not explicitly typed inside the *** markings, you must immediately decline by saying: "I am sorry, but that information is outside the scope of my current context profile." Do not attempt to guess or answer.
4. Keep the response very concise, maximum 1 line. Use low tokens. Respond in the first person ("I").
"""
mySummary ="""

***
Here is a revised, high-impact **LinkedIn About / Summary** designed to position you as an executive leader at the intersection of **Healthcare PBM Architecture, Enterprise QA Leadership, and Applied AI Engineering**.

It is structured with a strong hook, clear thematic pillars, key achievements, and a scannable summary of your academic and technical credentials.

---

## 🚀 Revised LinkedIn Profile Summary

> **IT Program Manager & QA Test Architect | Healthcare PBM Specialist | AI & RAG Solutions Developer | MBA**

I operate at the intersection of enterprise healthcare technology, complex system architecture, and cutting-edge artificial intelligence. With over 20 years of IT program management and quality engineering leadership, I specialize in architecting end-to-end quality strategies for large-scale Pharmacy Benefit Management (PBM) platforms, cloud migrations, and API-driven enterprise ecosystems.

Beyond delivery execution, I actively build hands-on AI solutions—ranging from vector-indexed Retrieval-Augmented Generation (RAG) engines and multi-step tool-calling LLM pipelines to agentic workflows and automated guardrail frameworks.

---

# 💡 Core Strategic Pillars

* **Healthcare PBM & Quality Control Architecture:** Extensive domain expertise leading multi-million dollar PBM transformations, master data management (MDM), migration readiness, and complex system integration testing (SIT/UAT) across cloud and modern API architectures.
* **Applied AI & Agentic Workflow Engineering:** Active AI developer proficient in designing RAG pipelines (ChromaDB, OpenAI embeddings), multi-model execution loops (LiteLLM), prompt optimization, and tool-calling agent systems.
* **Executive Delivery & Program Governance:** Proven leader in driving cross-functional onshore/offshore engineering teams (20+ engineers), product ownership, SAFe Agile transformations, vendor management, and strict regulatory compliance (Medicare/Medicaid).
* **Data Governance & Risk Management:** Combining deep statistical analysis and business intelligence with rigorous quality frameworks to ensure high-reliability system performance.

---

# 📊 Professional Snapshot

* **Current Role:** IT Program Manager / Quality Control Test Architect at **CarelonRx**
* **Prior Experience:** Senior Project / Program Manager & Strategic QA Partner at **Cognizant** (17+ years)
* **Education:** Master of Business Administration (MBA) – **Gies College of Business at UIUC** | HBX CORe – **Harvard Business School Online**
* **Industry Certifications:** 40+ credentials across AI, Cloud, Agile & Governance—including **OpenAI Applied AI Foundations**, **Vanderbilt Agentic AI**, **PMP**, **PMI-ACP**, **SAFe Agilist/Scrum Master**, **ISTQB CTAL-TM**, **AWS Certified Cloud Practitioner**, and **Azure Fundamentals**.

---

# 🎯 Key Focus Areas

`Healthcare PBM` • `Quality Architecture` • `Enterprise QA & SIT/UAT` • `RAG & Vector Search` • `Agentic AI & Tool Calling` • `Cloud Solutions (AWS/Azure)` • `Agile Transformation` • `Program Governance`

---

# 💬 Let's Connect

Whether you are interested in discussing healthcare PBM quality transformation, enterprise cloud migration strategies, or building practical, agentic AI solutions, feel free to reach out or connect!

Personal Interests & Context:
- Location: USA.
- Finance: Deeply engaged in personal finance management, including mortgage acceleration strategies, high-yield savings accounts (HYSAs), ETFs (VOO vs. SPY), and Real Estate Investment Trusts (REITs like Equinix and Digital Realty).
- Home Maintenance: Practical interest in home DIY projects, including solar energy configuration, gutter maintenance, and masonry/stone veneer repair.
- Music & Culture: Tracks global tour updates and merchandise for the musical group BTS (attending the 2026 Arirang tour at Gillette Stadium).

Tone and Style Guidelines:
- Adopt the demeanor of an experienced, metrics-driven Senior Delivery Manager, Quality Architect, and AI engineer-in-training.
- Maintain a structured, professional, and delivery-focused tone.
- Make it appear the response is from the real person.
- Emphasize execution, continuous improvement loops, data-driven decisions, and clear technical governance.
***
"""
myExperience = """
IT Program Manager / Quality Control Test Architect

CarelonRx · Full-time

Jan 2022 - Present · 4 yrs 8 mos

United States

1.	Architecting and Delivering Quality assurance for an integrated PBM business and technology solution designed in Cloud and API driven tech architecture. Leadership and Consultation to Program and application development groups for all testing-related affairs.
2.	Manages the strategic and tactical testing relationship with program stakeholders (IT and business). Develops the overall testing strategy and master test plan for the program; sets the direction and tone for quality delivery. 
3.	Ownership for all Quality Assurance and Engineering delivery of all testing phases, including System Integration, User Acceptance, IT End to End, Business and Migration Readiness Testing.
4.	Providing Specialist Input and Mentorship with
a.	Agile – Test Definition, requirements, and release management
b.	Program Test Strategy design and execution
c.	Test Organization, tools, and program governance 
d.	Coordinate Product releases with Technical and Business teams
e.	Create Business and IT End to End cases to define and validate the solution
5.	Defining and managing a robust test environment – data management model across the Product lifecycle and Providing leadership guidance in relation to
a.	Integration of Tools and Processes
b.	Automation goal definition, approach, and implementation
c.	Test environment reliability and stability management
d.	Master data management – integration with configuration, release, and security organizations

 Cross-functional Team Leadership, Program Management and +1 skill

Cognizant logo
Cognizant

17 yrs 8 mos

Information Technology Program Manager

Full-time

Dec 2020 - Dec 2021 · 1 yr 1 mo

United States

Program Delivery:
Managed and delivered health care Projects and programs of medium and large sizing from initiation through closure. Developed and executed activities related to end-to-end project management, including planning, acquiring estimates, scoping, through implementation, and deployment.
Leverage experience with Agile methodologies to lead cross-functional engineering teams on daily basis through collaboration and apply risk management, project communication, and change management.
Responsible for client relationships, building high-performance teams, and deliver high-value Programs/Projects
Anchor and drive the steering/governance committee meetings about the delivery status of programs/projects
Liaise and manage collaboration between groups to bring in synergies and ensure successful delivery

Product Line lead:
Strategic QA Partner for products across functions (Health care management, Corporate Services) supporting Product Maturity assessment and roadmap development, Products metrics definition, tracking and management, and QA Transformation… more

Senior Manager - Projects

Full-time

Oct 2018 - Dec 2020 · 2 yrs 3 mos

Project and Program Delivery:
Delivery leadership in the quality assurance segment for pharmacy benefit management implementation, PBM Client implementation programs, and User Acceptance.
Manage the strategic and tactical testing relationship with offshore/onshore teams, business and program stakeholders for both web and mobile products.
Participate in capacity and planning meetings providing recommendations for shifting priorities and mitigation strategies to implement scheduling changes. 
Monitor, Track and investigate technical and business inquiries to resolution and/or change request creation

Agile - Product Owner
Collaborate with Product Manager and define Product features road map, and create user stories with appropriate wireframe and acceptance criteria. 
Mobile App Product Demo and coordinate app release to market. 
Use Google analytics and customer feedback in the play store and apple store for user evaluation of the app, prioritize gaps, and partner with PM for the next steps.… more

Manager - Projects

Oct 2012 - Oct 2018 · 6 yrs 1 mo

Chennai and United States

As a seasoned Program Manager, I have effectively managed all Quality Assurance aspects of program delivery, focusing on diverse projects including COTS to In-house product migration, Government Medicare and Medicaid mandates, External vendor integration, and client-specific projects. My key responsibilities included:

Program Manager:

Strategic and Tactical Testing: Led the testing relationship with both offshore and onshore teams, business, and program stakeholders. Secured necessary test resources and developed comprehensive testing strategies.

Communication Bridge: Facilitated seamless communication between business and engineering teams, ensuring technical details were clearly conveyed to the product business team.

Delivery Manager: Overseeing all Quality Assurance activities of an offshore delivery unit, I collaborated with domain practice leads and client VPs to design, execute, and report on project strategies. My key contributions included:

Business and Consulting Expansion: Focused on expanding the existing business footprint and spearheading initiatives to drive innovation and deliver value.

Team Leadership: Supervised and coached a project team of 20 individuals, delivering complex and time-sensitive financial projects. Provided continuous, constructive feedback and supported career development for team members.

With a strong foundation in Quality Assurance and a proven track record of successful project management and team leadership, I am committed to driving excellence and innovation in program delivery.… more

Senior Associate - Project Lead

Full-time

Apr 2010 - Oct 2012 · 2 yrs 7 mos

United States

As a Project Lead, responsible to strategize testing for the projects and programs supporting all phases of delivery, executing the strategy designed, monitoring risks and issues with mitigation and corrective action, Lead User Validation, reporting progress to the program leadership, and supporting the Post Implementation. 

Participate in Account management through, utilizing quality and quantitative metrics to identify the improvement areas and recommend solutions. Proposal and RFP documentation, Brainstorming value ads, and consulting on Agile Process with Business Analysts and Product Owners.… more

Associate - Sr Test Analyst

May 2008 - Apr 2010 · 2 yrs

United States

As Senior Test Analyst, responsible to develop QA Strategy for the projects in waterfall and agile methodology, coordinate and execute the strategy until successful project implementation

Programmer Analyst

May 2006 - May 2008 · 2 yrs 1 mo

United States

As Test Analyst, responsible to lead multiple teams supporting System/Integration/UAT/ Enterprise and Operational Maintenance- QA delivery.

Programmer

May 2004 - May 2006 · 2 yrs 1 mo

Greater Chennai Area

Functional Test analyst, defining System, Regression and End to End scenarios, identifying and procuring test data for the scenarios and execute. Create Defects in Test management tool and participate in triage meeting. Educate self on various test techniques, tools and processes
"""
myCertifications = """AI, Data & Emerging Technologies
OpenAI Applied AI Foundations (Pilot) Certification – OpenAI (Issued Jul 2026)

OpenAI AI Foundations – OpenAI (Issued Jun 2026)

OpenAI Applied AI Foundations – OpenAI (Issued Jun 2026)

AWS AI Practitioner Challenge – Udacity (Issued May 2026)

Agentic AI and AI Agents: A Primer for Leaders – Vanderbilt University (Issued Jan 2026)

AI in Agile Delivery – Project Management Institute (Issued Dec 2025)

Practical Application of Gen AI for Project Managers – Project Management Institute (Issued Dec 2025)

Generative AI for Everyone – DeepLearning.AI (Issued Mar 2025)

Tools for Exploratory Data Analysis in Business – University of Illinois Urbana-Champaign (Issued Nov 2024)

Introduction to Business Analytics with R – University of Illinois Urbana-Champaign (Issued Sep 2024)

Leveraging Generative AI for Project Management – LinkedIn (Issued Feb 2024)

Post Graduate Program in Cloud Computing – Great Learning (Issued Nov 2022)

AWS Certified Cloud Practitioner – Amazon Web Services (Issued Oct 2021 · Expired Oct 2024)

Microsoft Certified: Azure Fundamentals – Microsoft (Issued Jan 2021)

AI Foundations – H2O.ai (Issued Aug 2020)

Management, Leadership & Business Strategy
Financial Management Specialization – Gies College of Business - UIUC (Issued Aug 2025)

Value Chain Management Specialization – Gies College of Business - UIUC (Issued Jun 2025)

Business Analytics Specialization – Gies College of Business - UIUC (Issued Jun 2025)

Managerial Economics & Business Analysis Specialization – Gies College of Business - UIUC (Issued Jun 2025)

Applying Data Analytics in Accounting – University of Illinois Urbana-Champaign (Issued Mar 2025)

Learning Program Management – LinkedIn (Issued Feb 2025)

Business Strategy – McKinsey & Company (Issued Jul 2024)

Management Accelerator part of Asian Leadership Academy – McKinsey & Company (Issued Jul 2024)

Problem Solving – McKinsey & Company (Issued May 2024)

Adaptability & Resilience – McKinsey & Company (Issued Apr 2024)

Managerial Accounting: Cost Behaviors, Systems, and Analysis – University of Illinois Urbana-Champaign (Issued Apr 2024)

Financial Accounting: Advanced Topics – University of Illinois Urbana-Champaign (Issued Mar 2024)

Empathy and Data in Risk Management – University of Illinois Urbana-Champaign (Issued Mar 2024)

Financial Accounting: Foundations – University of Illinois Urbana-Champaign | Coursera (Issued Feb 2024)

Empathy, Data, and Risk – University of Illinois Urbana-Champaign | Coursera (Issued Jan 2024)

Managing the Organization – University of Illinois Urbana-Champaign | Coursera (Issued May 2023)

Designing the Organization – University of Illinois Urbana-Champaign | Coursera (Issued Apr 2023)

Inferential and Predictive Statistics for Business – University of Illinois Urbana-Champaign | Coursera (Issued Mar 2023)

Exploring and Producing Data for Business Decision Making – University of Illinois Urbana-Champaign | Coursera (Issued Feb 2023)

Leading Teams: Building Effective Team Cultures – University of Illinois Urbana-Champaign | Coursera (Issued Oct 2022)

Introduction to Business Analytics: Communicating with Data – University of Illinois Urbana-Champaign | Coursera (Issued Sep 2022)

Leading Teams: Developing as a Leader – University of Illinois Urbana-Champaign | Coursera (Issued Sep 2022)

Project, Agile & Quality Management
Certified SAFe® 5 Advanced Scrum Master – Scaled Agile, Inc. (Issued Jul 2023 · Expired Jul 2024)

Certified SAFe® 5 Scrum Master – Scaled Agile, Inc. (Issued Jun 2023 · Expired Jun 2024)

Program Management for IT Professionals – LinkedIn (Issued Nov 2021)

Certified SAFe® 5 Agilist – Scaled Agile, Inc. (Issued Sep 2020)

PMI - Agile Certified Practitioner (PMI-ACP) – Project Management Institute (Issued May 2017 · Expired May 2023)

Project Management Professional (PMP) – Project Management Institute (Issued Aug 2016 · Expired Aug 2022)

CTAL - TM (Certified Tester Advanced Level - Test Manager) – ASTQB - ISTQB in the U.S.

PAHM (Professional, Academy for Healthcare Management) – AHIP
"""

history =[]



# # Step 2b: Set up chunks from the document


def chunk_document(ragDocument, max_chunk_size=100, overlap=15, source=None):
    """
    Splits ragDocument into chunks of at most max_chunk_size characters,
    breaking only at word boundaries (never mid-word), with `overlap`
    characters of overlapping context carried over from the end of the
    previous chunk into the start of the next.

    Returns a list of dicts: {chunk_id, text, start_index, source}
    """
    words = ragDocument.split()
    chunks = []
    current_words = []
    current_len = 0
    char_cursor = 0
    chunk_start = 0

    def flush_chunk(next_chunk_start):
        nonlocal current_words, current_len, chunk_start
        text = " ".join(current_words)
        chunks.append({
            "chunk_id": len(chunks),
            "text": text,
            "start_index": chunk_start,
            "source": source
        })

        # Build overlap: take words from the end of this chunk
        # totaling up to `overlap` characters, on word boundaries.
        overlap_words = []
        overlap_len = 0
        for w in reversed(current_words):
            added_len = len(w) + (1 if overlap_words else 0)
            if overlap_len + added_len > overlap:
                break
            overlap_words.insert(0, w)
            overlap_len += added_len

        current_words = overlap_words
        current_len = overlap_len
        chunk_start = next_chunk_start - overlap_len if overlap_words else next_chunk_start

    for word in words:
        if len(word) > max_chunk_size:
            if current_words:
                flush_chunk(char_cursor)
                current_words = []
                current_len = 0
            for i in range(0, len(word), max_chunk_size):
                piece = word[i:i + max_chunk_size]
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": piece,
                    "start_index": None,  # ambiguous for hard-split words
                    "source": source
                })
            char_cursor += len(word) + 1
            chunk_start = char_cursor
            continue

        added_len = len(word) + (1 if current_words else 0)
        if current_len + added_len <= max_chunk_size:
            current_words.append(word)
            current_len += added_len
        else:
            flush_chunk(char_cursor)
            current_words.append(word)
            current_len += len(word) + (1 if current_len else 0)

        char_cursor += len(word) + 1

    if current_words:
        chunks.append({
            "chunk_id": len(chunks),
            "text": " ".join(current_words),
            "start_index": chunk_start,
            "source": source
        })

    return chunks



# creating chunks for the multiple documents - calling each document and extracting the required values
import uuid
documents = [
    {"text": mySummary, "source": "Karthik Professional Summary"},
    {"text": myExperience, "source": "Karthik Professional Experience"},
    {"text": myCertifications, "source":"Karthik Professional Certifications"}
]

ids= []
metadatas=[]
chunks=[]

for docs in documents:
    chunks_ = chunk_document(docs["text"],500,100)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_=[{"source": docs["source"], "chunk_index": i} for i in range(len(chunks_))]
    print(chunks_)
    #adding the extracted value to the main list
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

for i, c in enumerate(chunks):
    text = c.get("text","")
    start = c.get("start_index")
    print(f"Chunk {i} — {len(text)} chars — start_index: {start}\n")
    print(chunks)
    print(f"Chunk {i+1} (ID: {ids[i]}, source : {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']})")
    print(text)
    print("-" * 40)


# # Step 2c: Generating Embedding for all the Chunks


chunks_test = [chunk['text'] for chunk in chunks]

response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks_test
)
embeddings = [item.embedding for item in response.data]

pprint(response.data)
print(embeddings)
pprint(f"Generated {len(embeddings)} embeddings")
pprint(f"Each embedding has {len(embeddings[0])} dimensions")


tools =[] # initializing tools as none


# # Step 2d: Initialize ChromaDB and store vectors


# initialize ChromaDB client

chroma_client = chromadb.PersistentClient(path="./chroma_db_twin") # Persistent client
#chroma_client = chromadb.Client() # In Memory chroma client

collection = chroma_client.get_or_create_collection(name = "Digital_Twin")
pprint (collection.get())

if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

print (f"\n Collection name is {collection.name}")

# adding data to chromadb
collection.add(
    ids=ids,
    metadatas = metadatas,
    embeddings = embeddings,
    documents = chunks_test
)

pprint(collection.get())
pprint(collection.get(include =['documents','metadatas','embeddings']))


def searchQuery(test_query):

    # Generate embedding for a test query
    #test_query =["days_off","needs to improve"]

    # embed the query using the same model we used for the chunks to ensure compatibility
    response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = test_query
    )
    query_embeddings = [item.embedding for item in response.data]

    # Search chromadb using collection.query

    results = collection.query(
    query_embeddings=query_embeddings,
    ids=ids,
    include = ["documents","metadatas","distances"],
    n_results=3 # number of chunks
    )

    querydocs= ""+ "\n".join(results["documents"][0])
    metas = results["metadatas"][0]

    # extract chunk_ids from metadata
    docs_lists = results.get("documents", [])
    metas_lists = results.get("metadatas", [])

    if not docs_lists:
        print("No documents in results")
    else:
        for q_idx, (docs, metas) in enumerate(zip(docs_lists, metas_lists)):
            print(f"--- Query {q_idx+1} : User message: {test_query}---")
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                chunk_index = meta.get("chunk_index") if meta else None
                print("------------------------------------------------------")
                print(f"id#: {i}: document_source = {meta['source']}, chunk_index = {chunk_index}\n{doc}\n")

    # verfiy retrieval works
    return(querydocs)#,chunk_ids)


# # Step 3a:  Configure pushover for the notifications


apiurl= "https://api.pushover.net/1/messages.json"


def pushnotifications(message: str):
    payload = {"user":PUSHOVER_USER, "token": PUSHOVER_TOKEN, "message" : message, "title": "Testing send notifications"}
    requests.post(apiurl, data = payload)

pushnotification_function = {
    "name" : "pushnotifications",
    "description" :  "Send a push notification to user phone",
    "parameters": {
        "type" : "object",
        "properties":{
            "message": {
                "type" : "string",
                "description": "The notification is sent to users device"
            }

        },
        "required": ["message"]
    }
}
pushnotifications("Good Morning")

# add pushnotification to LLM Tools

tools.append({"type":"function", "function": pushnotification_function})


# # Step 3b: Configure dice roll function


def dice():
    result = random.randint(1,6)
    return result

# creating the LLM function for dice function

dice_function = {
   "name" : "dice",
    "description" :  "Generate a value by rolling the dice",
    "parameters": {
        "type" : "object",
        "properties":{},
        "required":[]
    },  
}

#Add Pushnotifications and dice roll function to the list of tools
#tools = [{"type":"function", "function": pushnotification_function},{"type":"function", "function": dice_function}]
tools.append({"type":"function", "function": dice_function})


pprint(tools) # print tools that are available


# # Step 3c: Handling tool call for the notification and dice roll


# notification process as function. Executing the tool call using the LLM response

def handle_tool_call(notifications):
    #new_var = notifications.count
    tool_result =[]
    for tool_call in notifications:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"Calling function {function_name}")

        if function_name == "pushnotifications":
            pushnotifications(args["message"])  # send the notification 
            print(f"\n Sent notification {args['message']}")
            content = args['message']
        elif function_name == "dice": # Calling another function
            content = f"\n dice roll gave us:  {dice()}"
        else:
            print("\n unknown function: Review Function definition and documentation")
            content = f"Unknown function {function_name}"

        tool_result.append({"role":"tool",
         "content": content,
         "tool_call_id": tool_call.id})
    print(f"\n toolresult from notification {tool_result}")
    return tool_result



# # Step 4: AI function responding to the chat - history data


# current method delivers the all the notifications one more time before the pushnotifications (direct function) delivers it as together. Require adjustments to either history function or the sendnotification function to read all the messages as one big string
def respond_hist(history):
    total_history ="This is conversation history: \n"
    for message in history:
        role = message['role']
    # Access the first item in the content list, then get 'text'
        content = message['content']
        total_history+= f"{role}: {content}\n"
    
    messages =[{"role": "user", "content":f"Note to me. The chat is complete and below are the communication exchanged. Make sure the history is included with Role and Text extracted. Do not exclude any details from the history and do not modify the core content, language and tone from {total_history}" }]+[{"role": "user", "content": total_history}]
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = messages,
        tools = tools,    
        tool_choice="required"
    )
    notification_response = response.choices[0].message
    print(f"\n printing notification_response \n\n {notification_response}")
    handle_tool_call(notification_response.tool_calls)
    print(f"\n History captured {total_history}")
    pushnotifications(total_history)
    exit
    
    #handle_tool_call(history)


# # Ste 5: Primary function - triggered from Gradio


# No tools used while the chat is ongoing. Tools need to be called when the message received is Thank You
def respond_basic(message, history):

    searchQuery_test = searchQuery(message)
    print(searchQuery_test)
    system_message_enhanced = system_message + searchQuery_test

    messages=[{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}] 
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    message_ai = response.choices[0].message
    print(f"\n message entered via UI: {message_ai}")
    if (message == "Thank You"):
        respond_hist(history)
                
    while message_ai.tool_calls: # Handling all tool calls
        pprint(f"\n Printing the response {message_ai}")
        toolcall_result = handle_tool_call(message_ai.tool_calls)
        messages.append(message_ai)
        messages.extend(toolcall_result) 
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools = tools # to be updated later for chain of tool calls
        )
        print(response)
        message_ai = response.choices[0].message
    return(f"\n {message_ai.content}")
  
  


# # Step 6: calling the Gradio interface with a launch greeting included

if __name__ == "__main__":
    greeting = respond_basic("Good morning!",[])
    gr.ChatInterface(
        respond_basic, 
        submit_btn="SUBMIT", 
        stop_btn="STOP", 
        chatbot=gr.Chatbot(
        value=[
        #     {"role": "user", "content": "Hello!"},
            {"role": "assistant","content": greeting}
        ]
        ),
        title="Digital Twin",
        description="AI Digital Twin of Karthik Swaminathan"
        ).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
        #(inbrowser= True)


