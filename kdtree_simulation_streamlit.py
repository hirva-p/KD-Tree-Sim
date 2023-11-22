import streamlit as st
import numpy as np 
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt 
import random as rd


algo = st.selectbox("Select an algorithm:",("KD Tree", "LSH"))
if algo =="KD Tree":
    st.title("KD Tree") 
    if "cnt" not in st.session_state:
        st.session_state["cnt"] = 0

    if "input_nums_x" not in st.session_state:
        st.session_state.input_nums_x = []

    if "input_nums_y" not in st.session_state:
        st.session_state.input_nums_y = []

    if "input_num" not in st.session_state:
        st.session_state.input_num = []

    if "cnt2" not in st.session_state:
        st.session_state["cnt2"] = 0

    if "cnt3" not in st.session_state:
        st.session_state["cnt3"] = 0

    if "labels" not in st.session_state:
        st.session_state.labels = []

    if "inp_num_check" not in st.session_state:
        st.session_state.inp_num_check = True

    if "nn" not in st.session_state:
        st.session_state.nn = []

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

    # st.markdown(st.session_state.input_nums_x)
    # st.markdown(st.session_state.input_nums_y)


    with st.form("my_form_q"):
        q = st.text_input("Add a query point(space separated)", key="q")
        subm_q = st.form_submit_button("Submit")

    depthlst = {}

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
            depthlst[depth] = 1
            median = len(data) // 2
        
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

    def helper_traversal(root):
        lst = []
        if root is not None:
            lst.append(root.point)
            if root.left is not None:
                lst.extend(helper_traversal(root.left))
            if root.right is not None:
                lst.extend(helper_traversal(root.right))
        return lst

    poss3 = {}
    def dfs_traversal2(root,g, currdepth = 0):
        if root:
            strr = "Medians are:"
            if root.left:
                if (root.depth%2==0) and (currdepth>root.depth):
                    g.add_edges_from([(str(root.point)+' '+str(dimn[0]), str(root.left.point)+' '+dimn[1])])
                    lt = poss3[str(root.point)+' '+str(dimn[0])]
                    height = len(positions)
                    poss3[str(root.left.point)+' '+dimn[1]] = ((lt[2]+lt[0])/2,500 - 500*(root.left.depth)/height ,lt[2],lt[0])
                    dfs_traversal2(root.left, g,currdepth)
                elif (root.depth%2==0) and (currdepth==root.depth):
                    helplst = helper_traversal(root.left)
                    helplst = sorted(helplst, key=lambda x: x[1])
                    # st.write(helplst)
                    str2 = ""
                    for i in range(0,len(helplst)):
                        str2 +=" "+str(helplst[i])
                    strr+=" "+str(helplst[len(helplst)//2])
                    g.add_edges_from([(str(root.point)+' '+str(dimn[0]), str(str2)+' '+dimn[1])])
                    lt = poss3[str(root.point)+' '+str(dimn[0])]
                    height = len(positions)
                    poss3[str(str2)+' '+dimn[1]] = ((lt[2]+lt[0])/2,500 - 500*(root.left.depth)/height ,lt[2],lt[0])
                elif (root.depth%2==1) and (currdepth==root.depth):
                    helplst = helper_traversal(root.left)
                    helplst = sorted(helplst, key=lambda x: x[0])
                    strr+=" "+str(helplst[len(helplst)//2])
                    str2 = ""
                    for i in range(0,len(helplst)):
                        str2 +=" "+str(helplst[i])
                    # st.write("str2",str2)
                    g.add_edges_from([(str(root.point)+' '+str(dimn[1]), str(str2)+' '+dimn[0])])
                    lt = poss3[str(root.point)+' '+str(dimn[1])]
                    height = len(positions)
                    poss3[str(str2)+' '+dimn[0]] = ((lt[2]+lt[0])/2,500 - 500*(root.left.depth)/height ,lt[2],lt[0])
                elif (root.depth%2==1) and (currdepth>root.depth):
                    g.add_edges_from([(str(root.point)+' '+dimn[1], str(root.left.point)+' '+str(dimn[0]))])
                    lt = poss3[str(root.point)+' '+dimn[1]]
                    height = len(positions)
                    poss3[str(root.left.point)+' '+str(dimn[0])] = ((lt[2]+lt[0])/2,500 - 500*(root.left.depth)/height ,lt[2],lt[0])
                    # st.write("dfs_traversal2 left elif",root.left.point)
                    dfs_traversal2(root.left, g,currdepth)
            if root.right:
                if (root.depth%2==0)and (currdepth>root.depth):
                    g.add_edges_from([(str(root.point)+' '+str(dimn[0]), str(root.right.point)+' '+dimn[1])])
                    lt = poss3[str(root.point)+' '+str(dimn[0])]
                    height = len(positions)
                    poss3[str(root.right.point)+' '+dimn[1]] = ((lt[0]+lt[3])/2,500 - 500*(root.right.depth)/height ,lt[0],lt[3])
                    # st.write("dfs_traversal2 right",root.right.point)
                    dfs_traversal2(root.right, g,currdepth)
                elif ((root.depth%2==0) and (currdepth==root.depth)):
                    helplst = helper_traversal(root.right)
                    helplst = sorted(helplst, key=lambda x: x[1])
                    strr+=" "+str(helplst[len(helplst)//2])
                    str2 = ""
                    for i in range(0,len(helplst)):
                        str2 +=" "+str(helplst[i])
                    # st.markdown(str2)
                    g.add_edges_from([(str(root.point)+' '+str(dimn[0]), str(str2)+' '+dimn[1])])
                    lt = poss3[str(root.point)+' '+str(dimn[0])]
                    height = len(positions)
                    poss3[str(str2)+' '+dimn[1]] = ((lt[0]+lt[3])/2,500 - 500*(root.right.depth)/height ,lt[0],lt[3])
                elif ((root.depth%2==1) and (currdepth==root.depth)):
                    helplst = helper_traversal(root.right)
                    helplst = sorted(helplst, key=lambda x: x[0])
                    strr+=" "+str(helplst[len(helplst)//2])
                    str2 = ""
                    for i in range(0,len(helplst)):
                        str2 +=" "+str(helplst[i])
                    # st.markdown(str2)
                    g.add_edges_from([(str(root.point)+' '+dimn[1], str(str2)+' '+str(dimn[0]))])
                    lt = poss3[str(root.point)+' '+dimn[1]]
                    height = len(positions)
                    poss3[str(str2)+' '+str(dimn[0])] = ((lt[0]+lt[3])/2,500 - 500*(root.right.depth)/height ,lt[0],lt[3])
                else :
                    g.add_edges_from([(str(root.point)+' '+dimn[1], str(root.right.point)+' '+str(dimn[0]))])
                    lt = poss3[str(root.point)+' '+dimn[1]]
                    height = len(positions)
                    poss3[str(root.right.point)+' '+str(dimn[0])] = ((lt[0]+lt[3])/2,500 - 500*(root.right.depth)/height ,lt[0],lt[3])
                    # st.write("dfs_traversal2 right elif",root.right.point)
                    dfs_traversal2(root.right, g,currdepth)
            stream2.append(strr)
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
    maxdepth = len(depthlst)
    q = q.split()
    qq = [int(q[0]),int(q[1])]
    st.write("Input numbers are: "+str(st.session_state.input_num))
    st.write("Query point is: "+str(qq))
    dot = nx.DiGraph()
    poss[str(kd.root.point)+' '+str(dimn[0])] = (500,500,0,1000)
    dfs_traversal(kd.root, dot)
    dot.add_node(str(qq))
    dot2 = nx.DiGraph()
    stream2 = []
    def creation_sim(data, kdd,depth=0):
        if st.session_state["cnt3"]>=0 and st.session_state["cnt3"]<=maxdepth:
            poss3[str(kd.root.point)+' '+str(dimn[0])] = (500,500,0,1000)
            dfs_traversal2(kd.root,dot2, st.session_state["cnt3"])
            # st.write("cnt3", st.session_state["cnt3"])
            
        
    # st.markdown(dimn)

    creation_sim(ip,kd, st.session_state["cnt3"])
    # st.markdown(poss3)


    poss2 ={}
    poss2[str(qq)] = (700,500)
    for key in poss:
        val = poss[key]
        poss2[key] = (val[0],val[1])
    # st.markdown(poss2)
    poss4 ={}
    poss4[str(qq)] = (700,500)
    for key in poss3:
        val = poss3[key]
        poss4[key] = (val[0],val[1])
    # st.markdown(poss4)
    fig, ax = plt.subplots()
    fig2,ax2 = plt.subplots()



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
                    stream.append("Distance between "+str(node.point)+" is less than the last best distance."+str(int(distance(best[-1],target)))+" Thus we replace.")
                    poslst.append((poss2[str(node.point)+' '+dimn[axis]][0]+200, poss2[str(node.point)+' '+dimn[axis]][1]))
                    edgelst.append(str(str(node.point)+' '+dimn[axis]))
                    anotheredgelst.append(node.point)
                else :
                    stream.append("Distance between "+str(node.point)+" is less than the last best distance."+str(int(distance(best[-1],target)))+" Thus we replace.")
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
        st.session_state.nn = np.array(l)

    create_sim(qq)
    # st.balloons()

    option = st.selectbox("Select an option:",("Creation of KD Tree", "Finding Nearest Neighbors"))
    # st.write("You selected option:",option) 
    if (option=="Creation of KD Tree"):
        c1,c2 = st.columns(2)
        with c2: 
            if st.button('Next',key = 'r'):
                st.session_state["cnt3"]+=1
                st.experimental_rerun()
        with c1:
            if st.button('Prev',key = 'l'):
                st.session_state["cnt3"]-=1
                st.experimental_rerun()
        st.write(stream2[st.session_state["cnt3"]])
        nx.draw_networkx(dot2, pos=poss4,node_shape="s", bbox=dict(facecolor="white"), ax=ax2)
        lowx = 500
        lowy = 500
        highx = 500
        highy = 500
        for i in poss4:
            if (int(poss4[i][0])<lowx):
                lowx = int(poss4[i][0])
            if (int(poss4[i][0])>highx):
                highx = int(poss4[i][0])
            if (int(poss4[i][1])<lowy):
                lowy = int(poss4[i][1])
            if (int(poss4[i][1])>highy):
                highy = int(poss4[i][1])
        ax2.set_xlim(lowx-200,highx+200)
        ax2.set_ylim(lowy-50,highy+50)
        st.pyplot(fig2)
        
    elif (option=="Finding Nearest Neighbors"):
        c1,c2 = st.columns(2)
        with c2: 
            if st.button('Next',key = 'r'):
                st.session_state["cnt2"]+=1
                st.experimental_rerun()
        with c1:
            if st.button('Prev',key = 'l'):
                st.session_state["cnt2"]-=1
                st.experimental_rerun()
        if (st.session_state["cnt2"]>=0 & st.session_state["cnt2"]<len(stream)):
            poss2[str(qq)] = poslst[ st.session_state["cnt2"]]
            st.markdown(stream[st.session_state["cnt2"]])
            
            dot.add_edge(str(qq),edgelst[st.session_state["cnt2"]])
            labels[(str(qq),edgelst[st.session_state["cnt2"]])] = str(round(distance(qq,anotheredgelst[st.session_state["cnt2"]]),2))
            nx.draw_networkx(dot, pos=poss2,node_shape="s", bbox=dict(facecolor="white"), ax=ax)
            nx.draw_networkx_edge_labels(dot,pos=poss2,edge_labels=labels,ax =ax)
        lowx = 500
        lowy = 500
        highx = 500
        highy = 500
        for i in poss2:
            if (int(poss2[i][0])<lowx):
                lowx = int(poss2[i][0])
            if (int(poss2[i][0])>highx):
                highx = int(poss2[i][0])
            if (int(poss2[i][1])<lowy):
                lowy = int(poss2[i][1])
            if (int(poss2[i][1])>highy):
                highy = int(poss2[i][1])
        ax.set_xlim(lowx-100,highx+100)
        ax.set_ylim(lowy-50,highy+50)
        st.pyplot(fig)
        st.write("3 Nearest Neighbors are: "+str(st.session_state.nn))
        dist = []
        for i in st.session_state.nn:
            dist.append(round(distance(qq,np.array(i)),2))
        st.write("with distances: "+str(dist)+" respectively")

elif algo=="LSH":
    st.title("LSH")
    if "cnt" not in st.session_state:
        st.session_state["cnt"] = 0

    if "input_nums_x" not in st.session_state:
        st.session_state.input_nums_x = []

    if "input_nums_y" not in st.session_state:
        st.session_state.input_nums_y = []

    if "input_num" not in st.session_state:
        st.session_state.input_num = []

    if "nn_bin" not in st.session_state:
        st.session_state.nn_bin = []
    if "cnt2" not in st.session_state:
        st.session_state["cnt2"] = 1

    if "cnt3" not in st.session_state:
        st.session_state["cnt3"] = 0

    if "labels" not in st.session_state:
        st.session_state.labels = []

    if "inp_num_check" not in st.session_state:
        st.session_state.inp_num_check = True

    if "1" not in st.session_state:
        st.session_state["1"] = False
    if "2" not in st.session_state:
        st.session_state["1"] = False
    if "3" not in st.session_state:
        st.session_state["3"] = False 

    if "rand_num" not in st.session_state:
        st.session_state["rand_num"] = 0

    with st.form("rand_nums"):
        st.session_state["rand_num"] = int(st.number_input("Pick a random number"))
        subm_n = st.form_submit_button("Submit")
        if subm_n:
            st.session_state["1"] = False
            st.session_state["2"] = False
            st.session_state["3"] = False
            st.session_state.nn_bin = []

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

    # st.markdown(st.session_state.input_nums_x)
    # st.markdown(st.session_state.input_nums_y)

    with st.form("my_form_q"):
        q = st.text_input("Add a query point(space separated)", key="q")
        subm_q = st.form_submit_button("Submit")

    q = q.split()
    qq = [int(q[0]),int(q[1])]
    st.write("Query point is: "+str(qq))
    if (st.session_state.inp_num_check ==True and len(st.session_state.input_nums_y)==l):
        for i in range(0,len(st.session_state.input_nums_x)):
            x = int(st.session_state.input_nums_x[i])
            y = int(st.session_state.input_nums_y[i])
            st.session_state.input_num.append([x,y])
        
        st.session_state.inp_num_check = False
    ip = np.array(st.session_state.input_num)
    st.write("Input Numbers are: "+str(st.session_state.input_num))
    # plt.scatter(st.session_state.input_nums_x,st.session_state.input_nums_y)


    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title("LSH")
    ax.scatter(st.session_state.input_nums_x,st.session_state.input_nums_y, label='circle marker')
    lowx,lowy,highx,highy = min(st.session_state.input_nums_x), min(st.session_state.input_nums_y), max(st.session_state.input_nums_x), max(st.session_state.input_nums_y)

    annotate_text = []
    for i in st.session_state.input_num:
        annotate_text.append(str(i))
    # st.write(annotate_text)
    for i in range(len(annotate_text)): 
        ax.annotate(annotate_text[i],(st.session_state.input_nums_x[i]-0.25,st.session_state.input_nums_y[i]+0.2))

    def remove_duplicates(list_of_lists):
        seen = set()
        result = []
        for sublist in list_of_lists:
            # Convert the inner list to a tuple to make it hashable
            tuple_sublist = tuple(sublist)

            if tuple_sublist not in seen:
                seen.add(tuple_sublist)
                result.append(list(sublist))

        return result
    def show_graph(sed):
        rd.seed(sed)
        fpx = rd.randint(lowx,highx)
        fpy = rd.randint(lowy,highy)
        spx = rd.randint(1,7)
        spy = rd.randint(1,7)
        ax.axline((fpx,fpy),(spx,spy),color='grey')
        ffpx = rd.randint(lowx,highx)
        ffpy = rd.randint(lowy,highy)
        sspx = rd.randint(1,7)
        sspy = rd.randint(1,7)
        st.write(ffpx,ffpy,sspx,sspy)
        fffpx = rd.randint(lowx,highx)
        fffpy = rd.randint(lowy,highy)
        ssspx = rd.randint(1,7)
        ssspy = rd.randint(1,7)
        st.write(fffpx,fffpy,ssspx,ssspy)
        ax.axline((ffpx,ffpy),(sspx,sspy),color='green')
        ax.axline((fffpx,fffpy),(ssspx,ssspy),color='y')
        # ax.axline((0,seclinec),slope=seclinem,color='green')
        # ax.axline((0,thirdlinec),slope=firstlinem,color='green')
        st.pyplot(fig)

        dot = nx.DiGraph()
        fig2, ax2 = plt.subplots()
        poss= {}
        dot.add_edge(1,2)
        dot.add_edge(1,3)
        dot.add_edge(1,2)
        dot.add_edges_from([(2,4),(2,5),(3,6),(3,7)])
        dot.add_edges_from([(4,'000'),(4,'001'),(5,'010'),(5,'011'),(6,'100'),(6,'101'),(7,'110'),(7,'111')])
        poss[1]=[500,400]
        poss[2] = [250,250]
        poss[3] = [750,250]
        poss[4] = [100,100]
        poss[5] = [400,100]
        poss[6] = [600,100]
        poss[7] = [900,100]
        poss["000"] = [50,-50]
        poss["001"] = [150,-50]
        poss["010"] = [350,-50]
        poss["011"] = [450,-50]
        poss["100"] = [550,-50]
        poss["101"] = [650,-50]
        poss["110"] = [850,-50]
        poss["111"] = [950,-50]
        label = {}
        label[(1,2)] = 0
        label[(2,4)] = 0
        label[(3,6)] = 0
        label[(1,3)] = 1
        label[(2,5)] = 1
        label[(3,7)] = 1
        # nx.draw_networkx(dot,pos=poss,ax=ax2)
        c1,c2 = st.columns(2)
        with c2: 
            if st.button('Next',key = 'r'):
                st.session_state["cnt2"]+=1
                st.experimental_rerun()
        with c1:
            if st.button('Prev',key = 'l'):
                st.session_state["cnt2"]-=1
                st.experimental_rerun()
        color_label = ['green','grey','grey','y','y','y','y','b','b','b','b','b','b','b','b']

        st.write(st.session_state["cnt2"])

        stream = []

        for i in st.session_state.input_num:
            strr = ""
            kk = (ffpx)*(ffpy) - (sspx)*(sspy)
            ll = (i[0]-ffpx)*(sspy-ffpy) - (i[1]-ffpy)*(sspx-ffpx)
            if (ll<=0):
                strr = '0'
            else:
                strr = '1'
            l = (i[0]-fpx)*(spy-fpy) - (i[1]-fpy)*(spx-fpx)
            if (l>0):
                strr = strr+'1'
            else:
                strr = strr+'0'
            lll = (i[0]-fffpx)*(ssspy-fffpy) - (i[1]-fffpy)*(ssspx-fffpx)
            if (lll>0):
                strr = strr+'1'
            else:
                strr = strr+'0'
            stream.append(strr)
        # st.write(stream)
        outlst = {}
        for i in range(0,8):
            outlst[i] = []
        # st.write(int(stream[6],2))
        for i in range(len(stream)):
            # st.write(int(stream[i]))
            outlst[int(stream[i],2)].append(st.session_state.input_num[i])
        # st.write(outlst)

        for i in range(len(outlst)):
            if (i==0):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [50,-60-(50*(j+1))]
            elif (i==1):
                for j in range(len(outlst[i])):
                    # st.write(outlst[i][0])
                    poss[str(outlst[i][j])] = [150,-60-(50*(j+1))]
            elif (i==2):
                for j in range(len(outlst[i])):
                    # st.write(outlst[i][0])
                    poss[str(outlst[i][j])] = [350,-60-(50*(j+1))]
            elif (i==3):
                for j in range(len(outlst[i])):
                    # st.write(outlst[i][0])
                    poss[str(outlst[i][j])] = [450,-60-(50*(j+1))]
            elif (i==4):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [550,-60-(50*(j+1))]
            elif (i==5):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [650,-60-(50*(j+1))]
            elif (i==6):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [850,-60-(50*(j+1))] 
            elif (i==7):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [950,-60-(50*(j+1))] 

        if st.session_state["cnt2"]>=0 and st.session_state["cnt2"]<3*len(st.session_state.input_num):
            execnum = int(st.session_state["cnt2"]%3)
            num = int(st.session_state["cnt2"]/3)
            i = st.session_state.input_num[num]
            # st.write(num)
            for j in range(0,num):
                # st.write(str(st.session_state.input_num[j]))
                dot.add_node(str(st.session_state.input_num[j]))
                color_label.append('w')
            # st.write(i,execnum)
            # st.write(stream[num])
            stt = stream[num]
            if execnum==0:
                dot.add_edge(str(i),1)
                if stt[0]=='0': 
                    st.write("Point "+str(i)+" lies on the same side as origin.")
                    poss[str(i)] = [poss[1][0]-150,poss[1][1]]
                else :
                    st.write("Point "+str(i)+" lies on the opp side as origin.")
                    poss[str(i)] = [poss[1][0]+150,poss[1][1]]
            elif execnum==1:
                if stt[0]=='0':
                    dot.add_edge(str(i),2)
                    if stt[1]=='0': 
                        st.write("Point "+str(i)+" lies on the same side as origin.")
                        poss[str(i)] = [poss[2][0]-150,poss[2][1]]
                    else :
                        st.write("Point "+str(i)+" lies on the opp side as origin.")
                        poss[str(i)] = [poss[2][0]+150,poss[2][1]]
                else:
                    dot.add_edge(str(i),3)
                    if stt[1]=='0': 
                        st.write("Point "+str(i)+" lies on the same side as origin.")
                        poss[str(i)] = [poss[3][0]-150,poss[3][1]]
                    else :
                        st.write("Point "+str(i)+" lies on the opp side as origin.")
                        poss[str(i)] = [poss[3][0]+150,poss[3][1]]
            else:
                if stt[0]=='0':
                    if stt[1]=='0':
                        dot.add_edge(str(i),4)
                        if stt[2]=='0': 
                            st.write("Point "+str(i)+" lies on the same side as origin.")
                            poss[str(i)] = [poss[4][0]-150,poss[4][1]]
                        else :
                            st.write("Point "+str(i)+" lies on the opp side as origin.")
                            poss[str(i)] = [poss[4][0]+150,poss[4][1]]
                    else:
                        dot.add_edge(str(i),5)
                        if stt[2]=='0': 
                            st.write("Point "+str(i)+" lies on the same side as origin.")
                            poss[str(i)] = [poss[5][0]-150,poss[5][1]]
                        else :
                            st.write("Point "+str(i)+" lies on the opp side as origin.")
                            poss[str(i)] = [poss[5][0]+150,poss[5][1]]
                else:
                    if stt[1]=='0':
                        dot.add_edge(str(i),6)
                        if stt[2]=='0': 
                            st.write("Point "+str(i)+" lies on the same side as origin.")
                            poss[str(i)] = [poss[6][0]-150,poss[6][1]]
                        else :
                            st.write("Point "+str(i)+" lies on the opp side as origin.")
                            poss[str(i)] = [poss[6][0]+150,poss[6][1]]
                    else:
                        dot.add_edge(str(i),7)
                        if stt[2]=='0': 
                            st.write("Point "+str(i)+" lies on the same side as origin.")
                            poss[str(i)] = [poss[7][0]-150,poss[7][1]]
                        else :
                            st.write("Point "+str(i)+" lies on the opp side as origin.")
                            poss[str(i)] = [poss[7][0]+150,poss[7][1]]
            color_label.append('w')
        elif st.session_state["cnt2"]>=3*len(st.session_state.input_num):
            for j in range(0,len(st.session_state.input_num)):
                # st.write(str(st.session_state.input_num[j]))
                dot.add_node(str(st.session_state.input_num[j]))
                color_label.append('w')
        nx.draw_networkx(dot,pos=poss,ax=ax2,node_color=color_label,node_size=500,node_shape='s')
        st.pyplot(fig2)
        # st.balloons()

    def show_graph_q(sed,option):
        rd.seed(sed)
        fpx = rd.randint(lowx,highx)
        fpy = rd.randint(lowy,highy)
        spx = rd.randint(1,7)
        spy = rd.randint(1,7)
        ax.axline((fpx,fpy),(spx,spy),color='grey')
        ffpx = rd.randint(lowx,highx)
        ffpy = rd.randint(lowy,highy)
        sspx = rd.randint(1,7)
        sspy = rd.randint(1,7)
        st.write(ffpx,ffpy,sspx,sspy)
        fffpx = rd.randint(lowx,highx)
        fffpy = rd.randint(lowy,highy)
        ssspx = rd.randint(1,7)
        ssspy = rd.randint(1,7)
        st.write(fffpx,fffpy,ssspx,ssspy)
        ax.axline((ffpx,ffpy),(sspx,sspy),color='green')
        ax.axline((fffpx,fffpy),(ssspx,ssspy),color='y')
        # ax.axline((0,seclinec),slope=seclinem,color='green')
        # ax.axline((0,thirdlinec),slope=firstlinem,color='green')
        st.pyplot(fig)

        dot = nx.DiGraph()
        fig2, ax2 = plt.subplots()
        poss= {}
        dot.add_edge(1,2)
        dot.add_edge(1,3)
        dot.add_edge(1,2)
        dot.add_edges_from([(2,4),(2,5),(3,6),(3,7)])
        dot.add_edges_from([(4,'000'),(4,'001'),(5,'010'),(5,'011'),(6,'100'),(6,'101'),(7,'110'),(7,'111')])
        poss[1]=[500,400]
        poss[2] = [250,250]
        poss[3] = [750,250]
        poss[4] = [100,100]
        poss[5] = [400,100]
        poss[6] = [600,100]
        poss[7] = [900,100]
        poss["000"] = [50,-50]
        poss["001"] = [150,-50]
        poss["010"] = [350,-50]
        poss["011"] = [450,-50]
        poss["100"] = [550,-50]
        poss["101"] = [650,-50]
        poss["110"] = [850,-50]
        poss["111"] = [950,-50]
        label = {}
        label[(1,2)] = 0
        label[(2,4)] = 0
        label[(3,6)] = 0
        label[(1,3)] = 1
        label[(2,5)] = 1
        label[(3,7)] = 1
        # nx.draw_networkx(dot,pos=poss,ax=ax2)
        c1,c2 = st.columns(2)
        # with c2: 
        #     if st.button('Next',key = 'r'):
        #         st.session_state["cnt2"]+=1
        #         st.experimental_rerun()
        # with c1:
        #     if st.button('Prev',key = 'l'):
        #         st.session_state["cnt2"]-=1
                # st.experimental_rerun()
        color_label = ['green','grey','grey','y','y','y','y','b','b','b','b','b','b','b','b']

        st.write(st.session_state["cnt2"])

        stream = []

        for i in st.session_state.input_num:
            strr = ""
            kk = (ffpx)*(ffpy) - (sspx)*(sspy)
            ll = (i[0]-ffpx)*(sspy-ffpy) - (i[1]-ffpy)*(sspx-ffpx)
            if (ll<=0):
                strr = '0'
            else:
                strr = '1'
            l = (i[0]-fpx)*(spy-fpy) - (i[1]-fpy)*(spx-fpx)
            if (l>0):
                strr = strr+'1'
            else:
                strr = strr+'0'
            lll = (i[0]-fffpx)*(ssspy-fffpy) - (i[1]-fffpy)*(ssspx-fffpx)
            if (lll>0):
                strr = strr+'1'
            else:
                strr = strr+'0'
            stream.append(strr)
        # st.write(stream)
        outlst = {}
        for i in range(0,8):
            outlst[i] = []
        # st.write(int(stream[6],2))
        for i in range(len(stream)):
            # st.write(int(stream[i]))
            outlst[int(stream[i],2)].append(st.session_state.input_num[i])
        # st.write(outlst)

        for i in range(len(outlst)):
            if (i==0):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [50,-60-(50*(j+1))]
            elif (i==1):
                for j in range(len(outlst[i])):
                    # st.write(outlst[i][0])
                    poss[str(outlst[i][j])] = [150,-60-(50*(j+1))]
            elif (i==2):
                for j in range(len(outlst[i])):
                    # st.write(outlst[i][0])
                    poss[str(outlst[i][j])] = [350,-60-(50*(j+1))]
            elif (i==3):
                for j in range(len(outlst[i])):
                    # st.write(outlst[i][0])
                    poss[str(outlst[i][j])] = [450,-60-(50*(j+1))]
            elif (i==4):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [550,-60-(50*(j+1))]
            elif (i==5):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [650,-60-(50*(j+1))]
            elif (i==6):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [850,-60-(50*(j+1))] 
            elif (i==7):
                for j in range(len(outlst[i])):
                    poss[str(outlst[i][j])] = [950,-60-(50*(j+1))] 

        for j in range(0,len(st.session_state.input_num)):
            # st.write(str(st.session_state.input_num[j]))
            dot.add_node(str(st.session_state.input_num[j]))
            color_label.append('w')
        i = qq
        strrr = ""
        kk = (ffpx)*(ffpy) - (sspx)*(sspy)
        ll = (i[0]-ffpx)*(sspy-ffpy) - (i[1]-ffpy)*(sspx-ffpx)
        if (ll<=0):
            strrr = '0'
        else:
            strrr = '1'
        l = (i[0]-fpx)*(spy-fpy) - (i[1]-fpy)*(spx-fpx)
        if (l>0):
            strrr = strrr+'1'
        else:
            strrr = strrr+'0'
        lll = (i[0]-fffpx)*(ssspy-fffpy) - (i[1]-fffpy)*(ssspx-fffpx)
        if (lll>0):
            strrr = strrr+'1'
        else:
            strrr = strrr+'0'
        # stream.append(strrr)
        # st.write(strrr)
        st.write(str(qq)+' lies in the bin '+strrr)
        dot.add_node(str(qq))
        poss[str(qq)] = [50,500]
        color_label.append('r')
        st.write('Neighbors in this partition: '+str(outlst[int(strrr,2)])) 
        if option=="Partition 2" and st.session_state["2"]==False:
            st.session_state.nn_bin.extend(outlst[int(strrr,2)])
            st.session_state["2"] = True
        elif option=="Partition 1" and st.session_state["1"]==False:
            st.session_state.nn_bin.extend(outlst[int(strrr,2)])
            st.session_state["1"] = True
        elif option=="Partition 3" and st.session_state["3"]==False:
            st.session_state.nn_bin.extend(outlst[int(strrr,2)])
            st.session_state["3"] = True
        st.session_state.nn_bin = remove_duplicates(st.session_state.nn_bin)
        st.write("Global neighbors: "+str(st.session_state.nn_bin))
        nx.draw_networkx(dot,pos=poss,ax=ax2,node_color=color_label,node_size=500,node_shape='s')
        st.pyplot(fig2)
        # st.balloons()


    rd.seed(st.session_state["rand_num"])
    i = rd.randint(1,10)
    j = rd.randint(10,20)
    k = rd.randint(20,30)
    option = st.selectbox("Select an option:",("Partition 1", "Partition 2","Partition 3"))
    query =st.checkbox("With query point")
    if option=="Partition 1":
        if not query:
            show_graph(i)
        else: 
            show_graph_q(i,option)
    elif option=="Partition 2":
        if not query:
            show_graph(j)
        else: 
            show_graph_q(j,option)
    elif option=="Partition 3":
        if not query:
            show_graph(k)
        else: 
            show_graph_q(k,option)
    st.write(i,j,k)

    qq = np.array(qq)
    def distance(point):
        return np.linalg.norm(point-qq)
        
    if query:
        k = 3
        glob_nn = sorted(st.session_state.nn_bin,key=distance)
        glob_nn = glob_nn[:3]
        st.write("3 Nearest Neighbors are: "+str(glob_nn))


