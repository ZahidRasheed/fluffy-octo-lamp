import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import re
import os

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

st.set_page_config(layout="wide")

# Load data and create embeddings (as before)
df = pd.read_excel('data.xlsx')
model = SentenceTransformer('all-mpnet-base-v2')
property_embeddings = model.encode(df['PropertyName'].tolist())
trading_embeddings = model.encode(df['ListOrTradingAsName'].tolist())
df['property_embeddings'] = property_embeddings.tolist()
df['trading_embeddings'] = trading_embeddings.tolist()

def semantic_search(query, df, model, top_k=5):
    query_embedding = model.encode([query])
    property_similarities = cosine_similarity(query_embedding, np.array(df['property_embeddings'].tolist()))[0]
    trading_similarities = cosine_similarity(query_embedding, np.array(df['trading_embeddings'].tolist()))[0]
    combined_similarities = (property_similarities + trading_similarities) / 2
    top_indices = np.argsort(combined_similarities)[::-1][:top_k]
    return df.iloc[top_indices]

def identify_matches(query, df):
    property_names = df['PropertyName'].unique()
    trading_names = df['ListOrTradingAsName'].unique()

    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()

    matched_properties = [name for name in property_names if re.search(r'\b' + re.escape(name.lower()) + r'\b', query_lower, re.IGNORECASE)]
    matched_trading_names = [name for name in trading_names if re.search(r'\b' + re.escape(name.lower()) + r'\b', query_lower, re.IGNORECASE)]

    # Additional logic to handle queries with "total" or "recoveries"
    if "total" in query_lower or "recoveries" in query_lower:
        matched_properties += [name for name in property_names if "total" in name.lower() or "recoveries" in name.lower()]
        matched_trading_names += [name for name in trading_names if "total" in name.lower() or "recoveries" in name.lower()]

    return matched_properties, matched_trading_names

def filter_data(df, properties, trading_names):
    if properties:
        df = df[df['PropertyName'].isin(properties)]
    if trading_names:
        df = df[df['ListOrTradingAsName'].isin(trading_names)]
    return df

def process_query(query, df):
    matched_properties, matched_trading_names = identify_matches(query, df)
    filtered_df = filter_data(df, matched_properties, matched_trading_names)
    # Return only the original columns
    filtered_df = filtered_df[['PropertyName', 'ListOrTradingAsName', 'GLA (m²)', 'Rent', 'Recoveries', 'GMR', 'Rate (R/m²)']]
    return filtered_df

def create_prompt(query, data):
    data_str = data.to_string(index=False)
    prompt = """
    Data: {data_str} \n
    Query: {query} \n
    Do not show details of formula calculations, only the final results.
    Format currency to Rand and area to square meters.
    There are chances that the data might be missing, incomplete or irrelevant.
    If the query is about comparision, show various comparissions. 
    Use tabular format for the results.
    Find interesting insights based on the data and add a summary at the end of the response.
    Find news articles or reports related to the data and summarize them and show the link.
    Response should avoid duplication to have low token usage.
    """.format(query=query, data_str=data_str)
    return prompt

def send_to_chatgpt(prompt):
    # Call the OpenAI API
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=2000,
    n=1,
    stop=None,
    temperature=0.7)

    # Extract the response text
    answer = response.choices[0].message.content.strip()
    return answer

st.title("Semantic Search and AI Insights - Redefine")

# Sticky input section for the query at the top
query = st.text_input("", value="", key="main_query", label_visibility="visible", placeholder="Ask anything about the data...")
if query:
    filtered_df = process_query(query, df)
    with st.expander("Filtered Data"):
        st.dataframe(filtered_df, width=2000)
    prompt = create_prompt(query, filtered_df)
    

    # Show loading spinner while waiting for the response
    with st.spinner('Waiting for response...'):
         response = send_to_chatgpt(prompt)
         st.write(response)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Display the conversation history
response_container = st.container()
with response_container:
    for message in st.session_state.conversation_history:
        st.write(message)

# Add a collapsible section with example queries
with st.expander("Example Queries"):
    st.write("""
    - Give me the GLA of 1006 On the Lake for each listing.
    - How many spaces by American Swiss? What is average rent? What is average size rented. What is the average cost per square meter? How many spaces by Baby City? and do a comprehensive comparison among two of these.
    - List how many and all the tenants in 1006 On the Lake
    - Calculate the complete rent for 1006 On the Lake and show the percentage of each tenant.
    - Calculate the recoveries for 1006 On the Lake and show the percentage of each tenant.
    - List 1006 On the Lake by total recoveries
    - What is the average Rate (R/m²) for 1006 On the Lake
    - What is the average GMR for 1006 On the Lake
    - Comparing 1006 On the Lake vs 150 Rivonia Road in relation to the variables isolated.
    - List and count all the 150 Rivonia Road that each list rents space with Redefine Rate (R/m²)
    - What is the ACCUMULATIVE rent for Right To Care over all spaces rent and the total GLA rented.
    - What is the total recoveries for all spaces rented by Adidas
    - What is the total Rate (R/m²) for all spaces rented by American Swiss and how does it compare to other Baby City
    - What is the total rent for all spaces rented by American Swiss and how many spaces do they rent? Which is the most expensive space they rent and the least expensive
    - List GLA per Wembley 1
    - Total GLA per American Swiss as well as total rent, total recoveries and rate.
    - List all Recoveries by Wembley 1
    - List all Recoveries for American Swiss.
    - Recoveries listed by the Ushukela Industrial Park from most to least
    """)