
# coding: utf-8

# In[1]:


#Import List
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
#init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
import re
from sklearn.metrics import confusion_matrix
#import seaborn as sns
#import matplotlib.gridspec as gridspec
#from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
#from scipy.stats import multivariate_normal
#from sklearn.metrics import f1_score
#from sklearn.metrics import recall_score , average_precision_score
#from sklearn.metrics import precision_score, precision_recall_curve

#from sklearn.model_selection import KFold
#from sklearn.datasets import make_regression
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score

#%matplotlib inline


# In[2]:


#File 읽기
Region_df = pd.read_csv('C:/Users/Sangjin-Lee/salaries-by-region.csv')


# In[3]:


Region_df.head()


# In[4]:


#Region의 신입, 중간경력 연봉값 정규화
Region_df.fillna(0)

Region_df['Mid-Career Median Salary'] = Region_df['Mid-Career Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
Region_df['Mid-Career Median Salary'] = pd.to_numeric(Region_df['Mid-Career Median Salary'], errors='coerce')

Region_df['Starting Median Salary'] = Region_df['Starting Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
Region_df['Starting Median Salary'] = pd.to_numeric(Region_df['Starting Median Salary'], errors='coerce')

#Region의 경력직 25%연봉, 75%연봉값 정규화
Region_df['Mid-Career 25th Percentile Salary'] = Region_df['Mid-Career 25th Percentile Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
Region_df['Mid-Career 25th Percentile Salary'] = pd.to_numeric(Region_df['Mid-Career 25th Percentile Salary'], errors='ignore')

Region_df['Mid-Career 75th Percentile Salary'] = Region_df['Mid-Career 75th Percentile Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
Region_df['Mid-Career 75th Percentile Salary'] = pd.to_numeric(Region_df['Mid-Career 75th Percentile Salary'], errors='ignore')


# In[5]:


Region_df.head()


# In[6]:


# x축 : 대학교가 있는 지역, y축 : 중간경력의 연봉
# 박스 : 실수값 분포에서 1사분위수위 3사분위수, 박스의 가로선 : 중간값, 박스외부의선 : 3사분위와 1사분위의 수 사의이 거리의 1.5배길이
# 세로선 바깥쪽 점 : 아웃라이어(outlier)
f, ax = plt.subplots(figsize=(12, 5)) 
g = sns.boxplot(x=Region_df['Region'],y=Region_df['Mid-Career Median Salary'])
plt.show()


# In[7]:


inputdata1 = Region_df['Starting Median Salary']/100000
inputdata2 = Region_df['Mid-Career Median Salary']/1000000
inputdata3 = Region_df['Mid-Career 25th Percentile Salary']/100000
inputdata4 = Region_df['Mid-Career 75th Percentile Salary']/100000
output = Region_df['Region']


# In[8]:


inputdata = np.array([inputdata1,inputdata2,inputdata3,inputdata4])
inputdata = np.float32(inputdata)
inputdata.T
np.transpose(inputdata)
inputdata = np.swapaxes(inputdata,0,1)


# In[10]:


print(inputdata)


# In[11]:


sample = ['California', 'Western', 'Midwestern', 'Southern', 'Northeastern']
idx2string = list(set(sample))  # index -> string
string2idx = {c: i for i, c in enumerate(idx2string)}  # string -> idex


# In[12]:


print(sample)


# In[13]:


print(idx2string)


# In[14]:


print(string2idx)


# In[15]:


for i in range(len(output)):
    if(output[i]=='Southern'):
        output[i] = int(0)
    
    if(output[i]=='Western'):
        output[i] = int(1)
        
    if(output[i]=='Midwestern'):
        output[i] = int(2)
    
    if(output[i]=='California'):
        output[i] = int(3)

    if(output[i]=='Northeastern'):
        output[i] = int(4)


# In[16]:


output = np.array([output])
output.T
np.transpose(output)
output = np.swapaxes(output,0,1)

output = output.astype(str).astype(int)


# In[17]:


print(inputdata)


# In[18]:


learning_rate = 0.001


# In[19]:


nb_classes = 5 #0~4
X = tf.placeholder(tf.float32, [None,4])
Y = tf.placeholder(tf.int32, [None,1]) #0~4
Y_one_hot = tf.one_hot(Y, nb_classes) # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)


# In[20]:


W = tf.Variable(tf.random_normal([4,nb_classes]), name = 'Weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')


logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)

cost = tf.reduce_mean(cost_i)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[21]:


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[22]:


from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)

for train_index, test_index in cv.split(inputdata):
    print("test index :\n", test_index)
    print("." * 60 )        
    print("train index:\n", train_index)
    print("=" * 60 )


# In[23]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10000):
        sess.run(optimizer, feed_dict={X: inputdata[train_index], Y: output[train_index]})        
        
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: inputdata[train_index], Y: output[train_index]})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
    #Let's see if we can predict    
    pred = sess.run(prediction, feed_dict={X: inputdata[train_index]})
    loss, acc = sess.run([cost, accuracy], feed_dict={X: inputdata[test_index], Y: output[test_index]})
    print("Test_Acc: {:.2%}".format(acc))
    
  
  #      sess.run(optimizer, feed_dict={X: inputdata[test_index], Y: output[test_index]})
  #      loss, acc = sess.run([cost, accuracy], feed_dict=X: inputdata[test_index])
  #      print("Loss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
        
    #y_data: (N,1) = flatten => (N, ) matches pred.shape    
   # for p, y in zip(pred, output.flatten()): 
   #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


# In[24]:


#output = np.array([output])
#output.T
#np.transpose(output)
#output = np.swapaxes(output,0,1)

#output = output.astype(str).astype(int)


# In[25]:


############### Real Analysis ######################


# In[26]:


#File Read
College_df = pd.read_csv('C:/Users/Sangjin-Lee/salaries-by-college-type.csv')


# In[27]:


#College의 신입, 중간경력 연봉값 정규화
College_df['Mid-Career Median Salary'] = College_df['Mid-Career Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
College_df['Mid-Career Median Salary'] = pd.to_numeric(College_df['Mid-Career Median Salary'], errors='coerce')

College_df['Starting Median Salary'] = College_df['Starting Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
College_df['Starting Median Salary'] = pd.to_numeric(College_df['Starting Median Salary'], errors='coerce')

#College의 경력직 25%연봉, 75%연봉값 정규화
College_df['Mid-Career 25th Percentile Salary'] = College_df['Mid-Career 25th Percentile Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
College_df['Mid-Career 25th Percentile Salary'] = pd.to_numeric(College_df['Mid-Career 25th Percentile Salary'], errors='ignore')

College_df['Mid-Career 75th Percentile Salary'] = College_df['Mid-Career 75th Percentile Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
College_df['Mid-Career 75th Percentile Salary'] = pd.to_numeric(College_df['Mid-Career 75th Percentile Salary'], errors='ignore')


# In[28]:




fig, ax = plt.subplots(figsize=(15,10), ncols=3, nrows=1)

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .8     # the amount of width reserved for blank space between subplots
hspace =  1.5    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)

# The amount of space above titles
y_title_margin = 1.0

ax[0].set_title("Starting Median Salary", y = y_title_margin)
ax[1].set_title("How much you earn throughout", y = y_title_margin)
ax[2].set_title("How much you will reach after reaching 3rd quarter of your career", y = y_title_margin)

ax[0].set_xticklabels(College_df['School Type'], rotation='vertical', fontsize='large')
ax[1].set_xticklabels(College_df['School Type'], rotation='vertical', fontsize='large')
ax[2].set_xticklabels(College_df['School Type'], rotation='vertical', fontsize='large')
ax[0].set_ylim(35000,250000)
ax[1].set_ylim(35000,250000)
ax[2].set_ylim(35000,250000)

sns.boxplot(x='School Type',y='Starting Median Salary', data=College_df,ax=ax[0])
sns.boxplot(x='School Type',y='Mid-Career Median Salary', data=College_df,ax=ax[1])
sns.boxplot(x='School Type',y='Mid-Career 75th Percentile Salary', data=College_df,ax=ax[2])

plt.tight_layout()


College_df.head()

Cinputdata1 = College_df['Starting Median Salary']/10000
Cinputdata2 = College_df['Mid-Career Median Salary']/10000
Cinputdata3 = College_df['Mid-Career 25th Percentile Salary']/10000
Cinputdata4 = College_df['Mid-Career 75th Percentile Salary']/10000
Coutput = College_df['School Type']


# In[29]:


College_df.head()


# In[30]:


Cinputdata1 = College_df['Starting Median Salary']/10000
Cinputdata2 = College_df['Mid-Career Median Salary']/10000
Cinputdata3 = College_df['Mid-Career 25th Percentile Salary']/10000
Cinputdata4 = College_df['Mid-Career 75th Percentile Salary']/10000
Coutput = College_df['School Type']


# In[31]:


Cinputdata = np.array([Cinputdata1,Cinputdata2,Cinputdata3,Cinputdata4])
Cinputdata = np.float32(Cinputdata)
Cinputdata.T
np.transpose(Cinputdata)
Cinputdata = np.swapaxes(Cinputdata,0,1)


# In[32]:


print(Cinputdata)


# In[33]:


Csample = ['Engineering', 'Party', 'Liberal Arts', 'Ivy League', 'State']
Cidx2string = list(set(Csample))  # index -> string
Cstring2idx = {c: i for i, c in enumerate(Cidx2string)}  # string -> idex


# In[34]:


print(Csample)


# In[35]:


print(Cidx2string)


# In[36]:


print(Cstring2idx)


# In[37]:


for i in range(len(Coutput)):
    if(Coutput[i]=='Engineering'):
        Coutput[i] = int(0)
    
    if(Coutput[i]=='Ivy League'):
        Coutput[i] = int(1)
        
    if(Coutput[i]=='Party'):
        Coutput[i] = int(2)
    
    if(Coutput[i]=='Liberal Arts'):
        Coutput[i] = int(3)

    if(Coutput[i]=='State'):
        Coutput[i] = int(4)


# In[38]:


Coutput = np.array([Coutput])
Coutput.T
np.transpose(Coutput)
Coutput = np.swapaxes(Coutput,0,1)
Coutput = Coutput.astype(str).astype(int)


# In[39]:


print(Cinputdata)


# In[40]:


from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)

for train_index, test_index in cv.split(Cinputdata):
    print("test index :\n", test_index)
    print("." * 60 )        
    print("train index:\n", train_index)
    print("=" * 60 )


# In[41]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(15000):
        sess.run(optimizer, feed_dict={X: Cinputdata[train_index], Y: Coutput[train_index]})        
        
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: Cinputdata[train_index], Y: Coutput[train_index]})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
    #Let's see if we can predict    
    pred = sess.run(prediction, feed_dict={X: Cinputdata[train_index]})
    loss, acc = sess.run([cost, accuracy], feed_dict={X: Cinputdata[test_index], Y: Coutput[test_index]})
    print("Test_Acc: {:.2%}".format(acc))
    
  
  #      sess.run(optimizer, feed_dict={X: inputdata[test_index], Y: output[test_index]})
  #      loss, acc = sess.run([cost, accuracy], feed_dict=X: inputdata[test_index])
  #      print("Loss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
        
    #y_data: (N,1) = flatten => (N, ) matches pred.shape    
   # for p, y in zip(pred, output.flatten()): 
   #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


# In[42]:


###########################################################

