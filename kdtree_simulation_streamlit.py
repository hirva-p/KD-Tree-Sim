import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt 

st.session_state
if "cnt" not in st.session_state:
    st.session_state["cnt"] = 0

if "input_nums_x" not in st.session_state:
    st.session_state.input_nums_x = []

if "input_nums_y" not in st.session_state:
    st.session_state.input_nums_y = []

if "input_num" not in st.session_state:
    st.session_state.input_num = []

if "rerun" not in st.session_state:
    st.session_state.rerun = False

if "cnt2" not in st.session_state:
    st.session_state["cnt2"] = 0

if "labels" not in st.session_state:
    st.session_state.labels = []

if "inp_num_check" not in st.session_state:
    st.session_state.inp_num_check = True

with st.form("my_form_n"):
    numofinp = st.text_input("Add the number of inputs", key="n")
    subm_x = st.form_submit_button("Submit")

l = int(numofinp)
def show_form(i):
    with st.form("my_form_inp"+str(i)):
        c1, c2 = st.columns(2)
        with c1:
            nums_x = st.text_input("Add x input", key="x"+str(i))
        with c2:
            nums_y = st.text_input("Add y input", key="y"+str(i))
        subm_x = st.form_submit_button("Submit")
    if (subm_x):
        return int(nums_x),int(nums_y)
    else :
        st.markdown("submit")



if (st.session_state["cnt"]<l):
    x,y = show_form(st.session_state["cnt"])
    while (x is None):
        pass
    st.session_state.input_nums_x.append(x)
    st.session_state.input_nums_y.append(y)
    st.session_state["cnt"]= st.session_state["cnt"]+1
    st.session_state.rerun = True
    if (st.session_state.rerun==True):
        st.session_state.rerun = False
        st.experimental_rerun()

st.markdown(st.session_state.input_nums_x)
st.markdown(st.session_state.input_nums_y)

# with st.form("my_form_x"):
#     numsx = st.text_input("Add the x inputs (space separated)", key="x")
#     subm_x = st.form_submit_button("Submit")

# with st.form("my_form_y"):
#     numsy = st.text_input("Add the y inputs (space separated)", key="y")
#     subm_y = st.form_submit_button("Submit")

with st.form("my_form_q"):
    q = st.text_input("Add a query point(space separated)", key="q")
    subm_q = st.form_submit_button("Submit")

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
def find_index(node, positions):
    for d, lst in enumerate(positions):
        for ind, item in enumerate(lst):
            if (str(node)==str(item)):
                return ind, len(lst), len(positions)
    return -1, -1, -1

positions = []
poss = {}
dimn = ['x','y']
height = len(positions)

if (st.session_state.inp_num_check ==True and len(st.session_state.input_nums_y)==l):
    for i in range(0,len(st.session_state.input_nums_x)):
        x = int(st.session_state.input_nums_x[i])
        y = int(st.session_state.input_nums_y[i])
        st.session_state.input_num.append([x,y])
    
    st.session_state.inp_num_check = False

ip = np.array(st.session_state.input_num)
kd = KDTree4(ip)
q = q.split()
qq = [int(q[0]),int(q[1])]
st.markdown(st.session_state.input_num)
st.markdown(qq)
dot = nx.DiGraph()
poss[str(kd.root.point)+' '+str(dimn[0])] = (500,500,0,1000)
dfs_traversal(kd.root, dot)
dot.add_node(str(qq))


# st.markdown(dimn)

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
edgelst = []
anotheredgelst = []
labels = {}
k = 3
def closest_points(node, target, depth=0):
        if node is None:
            return []
        axis = depth % len(target)
        dim = dimn[axis]
        next_branch = None
        opposite_branch = None

        if target[axis] < node.point[axis]:
            dir = "left"
            next_branch = node.left
            opposite_branch = node.right
            stream.append(str(qq)+' lies to the left of '+str(node.point)+"so we move to the left branch.")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
            edgelst.append(str(str(node.point)+' '+dimn[axis]))
            anotheredgelst.append(node.point)
        else:
            dir = "right"
            next_branch = node.right
            opposite_branch = node.left
            stream.append(str(qq)+' lies to the right of '+str(node.point)+"so we move to the right branch.")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
            edgelst.append(str(str(node.point)+' '+dimn[axis]))
            anotheredgelst.append(node.point)

        best= closest_points(next_branch, target, depth + 1)
        if (next_branch==None and dir=="left"):
            stream.append("There is no left branch.")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
            edgelst.append(str(str(node.point)+' '+dimn[axis]))
            anotheredgelst.append(node.point)
        elif (next_branch==None and dir=="right"):
            stream.append("There is no right branch.")
            poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
            edgelst.append(str(str(node.point)+' '+dimn[axis]))
            anotheredgelst.append(node.point)

        if len(best) < k:
            best.append(node.point)
            best.sort(key=lambda x: distance(x, target))
            if (dir=="left"):
                stream.append("We add the "+str(node.point)+" to best list because we have less than "+str(k)+" points.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            elif (dir=="right"):
                stream.append("We add the "+str(node.point)+" to best list because we have less than "+str(k)+" points.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            # bsttmp = []
            # for t in best:
            #     bsttmp.append(t)
            # bestlst.append(bsttmp)
        elif distance(node.point, target) < distance(best[-1], target):
            best[-1] = (node.point)
            if (dir=="right"):
                stream.append("Distance between "+str(node.point)+" is less than the last best distance."+int(distance(best[-1],target))+" Thus we replace.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            else :
                stream.append("Distance between "+str(node.point)+" is less than the last best distance."+int(distance(best[-1],target))+" Thus we replace.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            # bsttmp = []
            # for t in best:
            #     bsttmp.append(t)
            # bestlst.append(bsttmp)
            best.sort(key=lambda x: distance(x, target))

        if (len(best)<k):
            if (dir=="right"):
                stream.append("We need more points so we go in opp branch")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            else :
                stream.append("We need more points so we go in opp branch")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            opp_best = closest_points(opposite_branch, target, depth + 1)
            if (dir=="left"):
                stream.append("There is no right branch.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            elif (dir=="left"):
                stream.append("There is no left branch.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            best.extend(opp_best)
            best.sort(key=lambda x: distance(x, target))
            best = best[:k]

        elif (target[axis] - node.point[axis]) ** 2 <= distance(best[-1], target):
            if(dir=="right"):
                stream.append("The perpendicular distance between "+str(node.point)+" is less than(or equal to) the last best distance found. So we go in opposite branch.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            else:
                stream.append("The perpendicular distance between "+str(node.point)+" is less than(or equal to) the last best distance found. So we go in opposite branch.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            opp_best = closest_points(opposite_branch, target, depth + 1)
            if (dir=="left"):
                stream.append("There is no right branch.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            elif (dir=="right"):
                stream.append("There is no left branch.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            best.extend(opp_best)
            best.sort(key=lambda x: distance(x, target))
            best = best[:k]
        else:
            if (dir=="right"):
                stream.append("We don't go in opp branch. We return.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            else:
                stream.append("We don't go in opp branch. We return.")
                poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]-200, poss2[str(node.point)+' '+dimn[axis]][1]))
                edgelst.append(str(str(node.point)+' '+dimn[axis]))
                anotheredgelst.append(node.point)
            
        return best
def create_sim(qr):
    l = closest_points(kd.root, qr)
    st.markdown("Final nearest neighbors are:")
    st.markdown(l)

create_sim(qq)
# st.markdown(stream)
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c4: 
    if st.button('Next'):
        st.session_state["cnt2"]+=1
with c3:
    if st.button('Prev'):
        st.session_state["cnt2"]-=1
if (st.session_state["cnt2"]>=0 & st.session_state["cnt2"]<len(stream)):
    poss2[str(qq)] = poslst[ st.session_state["cnt2"]]
    c1,c2,c3 = st.columns(3)
    with c2:
        st.markdown(stream[st.session_state["cnt2"]])
    # st.markdown(bestlst[st.session_state["cnt"]])
    # st.markdown(best)
    # if (st.session_state["cnt2"]>0):
    #     dot.remove_edges_from([(str(qq),edgelst[st.session_state["cnt2"]-1])])
    dot.add_edge(str(qq),edgelst[st.session_state["cnt2"]])
    labels[(str(qq),edgelst[st.session_state["cnt2"]])] = str(round(distance(qq,anotheredgelst[st.session_state["cnt2"]]),2))
    nx.draw_networkx(dot, pos=poss2,node_shape="s", bbox=dict(facecolor="white"), ax=ax)
    nx.draw_networkx_edge_labels(dot,pos=poss2,edge_labels=labels,ax =ax)
st.pyplot(fig)
st.markdown(stream)
