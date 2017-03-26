import pandas as pd
import os
import time
from datetime import datetime

from time import mktime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

style.use("dark_background")
path = "/Users/sunciao/Documents/dl/isagent"

def clean_data():
    df = pd.DataFrame(columns=['User',
                               'detail',
                               'appointment',
                               'tag'])

    agent_df = pd.DataFrame.from_csv("listanddetailraw.csv")
    all_user = agent_df.user1
    all_user2 = agent_df.user2
    all_user3 = agent_df.user3
    all_user4 = agent_df.user4
    all_user5 = agent_df.user5
    all_user6 = agent_df.user6
    all_user7 = agent_df.user7
    all_user8 = agent_df.user8
    all_user9 = agent_df.user9
    all_agent = agent_df.agent
    arange = range(1, agent_df.user1.size+1)
    #correctap = []
    #listPV0 =[]
    #listArea0 = []
    #listBudget0 = []
    #listRoom0 = []
    #listSquare0 = []
    #listAge0 = []
    #listTag0 = []
    #detail0 = []
    #appointment0 = []
    print(arange)
    for ncount in arange:
        #targetIndex = 0
        userStr = all_user.loc[ncount]
        #appointNo = 0
        areaNo = 0
        for ncount2 in arange:
            if all_user2.loc[ncount2] == userStr:
                print("matched user2",ncount,"with",ncount2)
                areaNo += agent_df.listArea.loc[ncount2]
        #print(appointNo)
        #listArea0.append(areaNo)
        agent_df.listArea0.loc[ncount] = areaNo
        print("areaNo",areaNo,"check",agent_df.listArea0.loc[ncount],"on count",ncount)

        budgetNo = 0
        for ncount3 in arange:
            if all_user3.loc[ncount3] == userStr:
                print("matched user3",ncount,"with",ncount3)
                budgetNo += agent_df.listBudget.loc[ncount3]
        agent_df.listBudget0.loc[ncount] = budgetNo
        print("budgetNo",budgetNo,"check",agent_df.listBudget0.loc[ncount],"on count",ncount)

        roomNo = 0
        for ncount4 in arange:
            if all_user4.loc[ncount4] == userStr:
                print("matched user4",ncount,"with",ncount4)
                roomNo += agent_df.listRoom.loc[ncount4]
        agent_df.listRoom0.loc[ncount] = roomNo
        print("roomNo",roomNo,"check",agent_df.listRoom0.loc[ncount],"on count",ncount)

        sqNo = 0
        for ncount5 in arange:
            if all_user5.loc[ncount5] == userStr:
                print("matched user5",ncount,"with",ncount5)
                sqNo += agent_df.listSquare.loc[ncount5]
        agent_df.listSquare0.loc[ncount] = sqNo
        print("sqNo",sqNo,"check",agent_df.listSquare0.loc[ncount],"on count",ncount)

        ageNo = 0
        for ncount6 in arange:
            if all_user6.loc[ncount6] == userStr:
                print("matched user6",ncount,"with",ncount6)
                ageNo += agent_df.listAge.loc[ncount6]
        agent_df.listAge0.loc[ncount] = ageNo
        print("ageNo",ageNo,"check",agent_df.listAge0.loc[ncount],"on count",ncount)

        tagNo = 0
        for ncount7 in arange:
            if all_user7.loc[ncount7] == userStr:
                print("matched user7",ncount,"with",ncount7)
                tagNo += agent_df.listTag.loc[ncount7]
        agent_df.listTag0.loc[ncount] = tagNo
        print("tagNo",tagNo,"check",agent_df.listTag0.loc[ncount],"on count",ncount)

        detailNo = 0
        for ncount8 in arange:
            if all_user8.loc[ncount8] == userStr:
                print("matched user8",ncount,"with",ncount8)
                detailNo += agent_df.detail.loc[ncount8]
        agent_df.detail0.loc[ncount] = detailNo
        print("detailNo",detailNo,"check",agent_df.detail0.loc[ncount],"on count",ncount)

        apMo = 0
        for ncount9 in arange:
            if all_user9.loc[ncount9] == userStr:
                print("matched user9",ncount,"with",ncount9)
                apMo += agent_df.appointment.loc[ncount9]
        agent_df.appointment0.loc[ncount] = apMo
        print("apMo",apMo,"check",agent_df.appointment0.loc[ncount],"on count",ncount)

        for ncountAg in arange:
            if all_agent.loc[ncountAg] == userStr:
                print("agent",ncount,"with",ncountAg)
                agent_df.tag.loc[ncount] = 1

    print(agent_df)

    agent_df.to_csv('datacleaned0318.csv')
        #time.sleep(15)


    print("agent_df:",all_user.loc[12])


clean_data()