import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Simple Test")
st.write("If you see this, Streamlit is working!")

# Test basic chart
data = {'x': [1, 2, 3], 'y': [1, 4, 2]}
df = pd.DataFrame(data)
fig = px.line(df, x='x', y='y', title='Test Chart')
st.plotly_chart(fig)