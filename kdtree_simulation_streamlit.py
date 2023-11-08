import streamlit as st
import graphviz as gr
import numpy as np
import pandas as pd
from PIL import Image
import io
import networkx as nx 
import matplotlib.pyplot as plt 

st.session_state
if "cnt" not in st.session_state:
    st.session_state["cnt"] = 0


with st.form("my_form_x"):
    numsx = st.text_input("Add the x inputs (space separated)", key="x")
    subm_x = st.form_submit_button("Submit")

with st.form("my_form_y"):
    numsy = st.text_input("Add the y inputs (space separated)", key="y")
    subm_y = st.form_submit_button("Submit")

with st.form("my_form_q"):
    q = st.text_input("Add a query point(space separated)", key="q")
    subm_q = st.form_submit_button("Submit")

nums_x = numsx.split()
nums_y = numsy.split()
input_num = []
output_num = []
for i in range(0,len(nums_x)):
    x = int(nums_x[i])
    y = int(nums_y[i])
    input_num.append([x,y])
q = q.split()
qq = [int(q[0]),int(q[1])]
st.markdown(input_num)
st.markdown(qq)

positions = []

class KDTree4:
    def __init__(self, data):
        self.root = self.build_kdtree(data)
        self.tree = None

    def build_kdtree(self, data, depth=0,pos=0):
        if len(data)==0:
            return None

        k = data.shape[1]
        axis = depth % k
        sorted_indices = np.argsort(data[:, axis])

        data = data[sorted_indices]
        

        median = len(data) // 2
       # node = Node(float(round(data.iloc[median][axis],3)))
        if len(positions)>depth:
            positions[depth].append(data[median])
        else:
            while len(positions)<=depth:
                positions.append([])
            positions[depth].append(data[median])
        return Node(
            point=data[median],
            left=self.build_kdtree(data[:median], depth + 1), 
            right=self.build_kdtree(data[median + 1:],  depth + 1),
            depth = depth
        )

class Node:
    def __init__(self, point, left=None, right=None, depth=0):
        self.point = point
        self.left = left
        self.right = right
        self.depth = depth

def find_index(node, positions):
    for d, lst in enumerate(positions):
        for ind, item in enumerate(lst):
            if (str(node)==str(item)):
                return ind, len(lst), len(positions)
    return -1, -1, -1

poss = {}
dimn = ['x','y']
height = len(positions)
def dfs_traversal(root, g):
    if root:
        if root.left:
            if (root.depth%2==0):
                g.add_edges_from([(str(root.point)+' '+str(dimn[0]), str(root.left.point)+' '+dimn[1])])
                lt = poss[str(root.point)+' '+str(dimn[0])]
                height = len(positions)
                poss[str(root.left.point)+' '+dimn[1]] = ((lt[2]+lt[0])/2,500 - 500*(root.left.depth)/height ,lt[2],lt[0])
            else :
                g.add_edges_from([(str(root.point)+' '+dimn[1], str(root.left.point)+' '+str(dimn[0]))])
                lt = poss[str(root.point)+' '+dimn[1]]
                height = len(positions)
                poss[str(root.left.point)+' '+str(dimn[0])] = ((lt[2]+lt[0])/2,500 - 500*(root.left.depth)/height ,lt[2],lt[0])
            dfs_traversal(root.left, g)
        if root.right:
            if (root.depth%2==0):
                g.add_edges_from([(str(root.point)+' '+str(dimn[0]), str(root.right.point)+' '+dimn[1])])
                lt = poss[str(root.point)+' '+str(dimn[0])]
                height = len(positions)
                poss[str(root.right.point)+' '+dimn[1]] = ((lt[0]+lt[3])/2,500 - 500*(root.right.depth)/height ,lt[0],lt[3])
            else :
                g.add_edges_from([(str(root.point)+' '+dimn[1], str(root.right.point)+' '+str(dimn[0]))])
                lt = poss[str(root.point)+' '+dimn[1]]
                height = len(positions)
                poss[str(root.right.point)+' '+str(dimn[0])] = ((lt[0]+lt[3])/2,500 - 500*(root.right.depth)/height ,lt[0],lt[3])
            dfs_traversal(root.right, g)

ip = np.array(input_num)
# st.markdown(dimn)
kd = KDTree4(ip)

dot = nx.DiGraph()
poss[str(kd.root.point)+' '+str(dimn[0])] = (500,500,0,1000)
dfs_traversal(kd.root, dot)
dot.add_node(str(qq))
# st.markdown(poss)
poss2 ={}
poss2[str(qq)] = (700,500)
for key in poss:
    val = poss[key]
    poss2[key] = (val[0],val[1])
# st.markdown(poss2)

fig, ax = plt.subplots()



oplst = []
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
# st.image(image, caption='gr Graph', use_column_width=True)

stream = []
poslst = []
bestlst = []
k = 3
def closest_points(node, target, depth=0):
        if node is None:
            return []
        axis = depth % len(target)
        dim = dimn[axis]
        next_branch = None
        opposite_branch = None

        if target[axis] < node.point[axis]:
            next_branch = node.left
            opposite_branch = node.right
            stream.append(str(qq)+' lies to the left of '+str(node.point))
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
        else:
            next_branch = node.right
            opposite_branch = node.left
            stream.append(str(qq)+' lies to the right of '+str(node.point))
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))

        best= closest_points(next_branch, target, depth + 1)
    
        if len(best) < k:
            best.append(node.point)
            best.sort(key=lambda x: distance(x, target))
            stream.append("We add the "+str(node.point)+" to best list")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
            # bsttmp = []
            # for t in best:
            #     bsttmp.append(t)
            # bestlst.append(bsttmp)
        elif distance(node.point, target) < distance(best[-1], target):
            best[-1] = (node.point)
            stream.append("Dist between "+str(node.point)+" is less than the last best distance. Thus we replace.")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
            # bsttmp = []
            # for t in best:
            #     bsttmp.append(t)
            # bestlst.append(bsttmp)
            best.sort(key=lambda x: distance(x, target))
            
        if (target[axis] - node.point[axis]) ** 2 <= distance(best[-1], target):
            stream.append("The perpendicular dist between "+str(node.point)+" is less than(or equal to) the last best distance found. So we go in opposite branch.")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
            opp_best = closest_points(opposite_branch, target, depth + 1)
            best.extend(opp_best)
            best.sort(key=lambda x: distance(x, target))
            best = best[:k]
        else:
            stream.append("We don't go in opp branch.")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
            
        return best
def create_sim(qr):
    l = closest_points(kd.root, qr)
    st.markdown(l)


create_sim(qq)
# st.markdown(stream)

if st.button('Next'):
    st.session_state["cnt"]+=1
if st.button('Prev'):
    st.session_state["cnt"]-=1
if (st.session_state["cnt"]>=0 & st.session_state["cnt"]<len(stream)):
    poss2[str(qq)] =poslst[ st.session_state["cnt"]]
    st.markdown(stream[st.session_state["cnt"]])
    # st.markdown(bestlst[st.session_state["cnt"]])
    # st.markdown(best)
    nx.draw_networkx(dot, pos=poss2,node_shape="s", bbox=dict(facecolor="white"), ax=ax)
st.pyplot(fig)
st.balloons()





     
    

