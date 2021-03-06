import numpy as np
import pandas as pd
import plotly.graph_objects as go
from quant_reg_funcs import ols_reg, quant_reg
import streamlit as st
from PIL import Image
import os
import re
google_analytics_js ="""
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-175428475-2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-175428475-2');
</script>
"""

#setting favicon
favicon = Image.open('illustration_resources/favicon.jpg')
st.set_page_config(page_title='CII Tool', page_icon=favicon)

a=os.path.dirname(st.__file__)+'/static/index.html'
with open(a, 'r') as f:
    data=f.read()
    if len(re.findall('UA-', data))==0:
        with open(a, 'w') as ff:
            newdata=re.sub('<head>','<head>'+google_analytics_js,data)
            ff.write(newdata)

from layout import _max_width_
_max_width_()

data = pd.read_csv('MRV_data.csv')
#data = data[(data['Ship type']=='Ro-ro ship')]#(data['Ship type']=='Container/ro-ro cargo ship')]#(data['Ship type']=='Container/ro-ro cargo ship')]#|(data['Ship type']=='Ro-ro ship')]

##hide menu and footer
hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

##Sidebar : Vessel Information
sidebarTitleTop= '<p style="font-weight: 700; color:#146EA6; font-size: 24px;">Graph Information</p>'
st.sidebar.markdown(sidebarTitleTop, unsafe_allow_html=True)

graph_type = st.sidebar.selectbox('CII Type',('AER','cgDist'))

if graph_type == 'AER':
    data['DWT'] = data['Deadweight']#GT']#
    data['AER'] = data['AER']#cgDIST']
    xaxis_label = 'DWT'
    yaxis_label = 'AER'
    lines = pd.read_csv('AER_lines.csv')

elif graph_type == 'cgDist':
    data['DWT'] = data['GT']#
    data['AER'] = data['cgDIST']
    xaxis_label = 'GT'
    yaxis_label = 'cgDist'
    lines = pd.read_csv('cgDist_lines.csv')

data['LOG_AER'] = np.log(data.AER)
data['LOG_DWT'] = np.log(data.DWT)

cIIHeader= '<p style="font-weight: 100; color:#042E43; font-size: 24px; margin-botttom: 0; ">CII Rating Boundaries and Quantile Regressions for '+graph_type+' (Prototype) </p>'
st.write(cIIHeader, unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=np.exp(data.LOG_DWT),
                        y=np.exp(data.LOG_AER),
                        text=data.Name,
                        mode='markers',
                        name='Data'))
x_range = np.arange(min(data.LOG_DWT),max(data.LOG_DWT),step=(max(data.LOG_DWT)-(min(data.LOG_DWT)))/100)
for letter, i in [('a',0.15),('b',0.35),('c',0.5),('d',0.65),('e',0.85)]:#[0.15,0.35,0.5,0.65,0.85]:
    if i == 0.5:
        fig.add_trace(go.Scatter(x=lines.x,
                                y=lines['quantile_0.5'],
                                mode='lines',
                                name='50% quantile (median)',
                                line=dict(color='black', dash='dot', width= 2)))
    else:
        if i == 0.15:
            fig.add_trace(go.Scatter(x=lines.x,
                                    y=lines['rating_'+letter],
                                    mode='lines',
                                    name='Rating Boundaries',
                                    opacity=0.4,
                                    line=dict(color='green', dash='dash', width= 2)))
            fig.add_trace(go.Scatter(x=lines.x,
                                    y=lines['quantile_'+str(i)],
                                    mode='lines',
                                    name='Quantile Regressions',
                                    line=dict(color='black', dash='solid', width= 2)))
        else:
            fig.add_trace(go.Scatter(x=lines.x,
                                    y=lines['rating_'+letter],
                                    mode='lines',
                                    name='Rating Boundaries',
                                    showlegend=False,
                                    opacity=0.4,
                                    line=dict(color='green', dash='dash', width= 2)))
            fig.add_trace(go.Scatter(x=lines.x,
                                    y=lines['quantile_'+str(i)],
                                    mode='lines',
                                    name='Quantile Regressions',
                                    showlegend=False,
                                    line=dict(color='black', dash='solid', width= 2)))
fig.update_layout(
    showlegend = True,
    xaxis_title=xaxis_label,
    yaxis_title=yaxis_label,
    width=1000,
    height=800,
    title='Ro-ro (no con-ros)'
    )
st.plotly_chart(fig, use_container_width = True)
st.write('Powered by ')
st.image('illustration_resources/arcsilea.png', width=200)
