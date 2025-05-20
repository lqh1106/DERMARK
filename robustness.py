import math
import random
import numpy as np
import scipy.stats as stats
def dynamic_programming_segment(P, bit,alpha, group_count,delta,beta):

    n = len(P)
    ed = pow(math.e, delta)
    alpha2 = pow(alpha,2)
    zpow = pow(stats.norm.ppf(1-alpha),2)
    
    # 预处理子段代价
    cost = [[float('inf')] * (n+1) for _ in range(n)]
    offsets = [[float('inf')] * (n+1) for _ in range(n)]
    red_count = [0] * (n + 1)
    green_count = [0] * (n + 1)
    for i in range(1, n + 1):
        red_count[i] = red_count[i - 1] + (1 if bit[i - 1] == 0 else 0)
        green_count[i] = green_count[i - 1] + (1 if bit[i - 1] == 1 else 0)

    right = [[float('inf')] * (n+1) for _ in range(n)]
    var = [[float('inf')] * (n+1) for _ in range(n)]
    offsets = [[float('inf')] * (n+1) for _ in range(n)]
    color_cost=[[float('inf')] * (n+1) for _ in range(n)]
    p_pre=[[float('inf')] * (n+1) for _ in range(n)]
    
    for start in range(n):
        for end in range(start+1, n+1):
            red = red_count[end-1]- red_count[start]
            green = green_count[end-1]- green_count[start]
            sum_p2 = 0
            sum_p1 = 0
            for k in range(start,end):
                p_pre = (green+0.5)*ed*P[k]/((red+green+1)*(ed*P[k]+1-P[k]))+(red+0.5)*ed*(1-P[k])/((red+green+1)*(ed*(1-P[k])+P[k]))
                sum_p2 += pow(p_pre, 2)
                sum_p1 += p_pre
            
            N = end - start
            right[start][end] = pow((sum_p1- 0.5*N), 2)/(sum_p1 - sum_p2)
            p_pre= sum_p1/N
            var[start][end] = (sum_p1 - sum_p2)/(N**2)
            red_in_segment = red_count[end] - red_count[start]
            green_in_segment = green_count[end] - green_count[start]
            
            color_cost[start][end] = min(-(red_in_segment*math.log(p_pre)+green_in_segment*math.log(1-p_pre)),-(red_in_segment*math.log(1-p_pre)+green_in_segment*math.log(p_pre)))
            
            # color_cost[start][end] = min(red_in_segment,green_in_segment)/N


    

    TIME = 0
    min_wc,min_cc = 0, 0
    mean_wc, mean_cc = 0, 0
    std_wc, std_cc = 1, 1
    while( TIME < 20):
        TIME +=1

        for start in range(n):
            
            for end in range(start+1, n+1):
                
                weight_cost = (zpow-right[start][end])**2
                wc = (weight_cost - min_wc) / (std_wc) 
                cc = (color_cost[start][end] - min_cc) / (std_cc) 
                
                cost[start][end] = beta* wc + cc

        # 动态规划表
        dp = [[float('inf')] * (group_count + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        path = [[-1] * (group_count + 1) for _ in range(n + 1)]  # 用于记录路径
        
        for j in range(1, group_count + 1):  # 枚举分段数
            for i in range(1, n + 1):       # 枚举小球位置
                for p in range(i):          # 枚举前一段的结束位置
                    if dp[p][j-1] + cost[p][i] < dp[i][j]:
                        dp[i][j] = dp[p][j-1] + cost[p][i]
                        path[i][j] = p
    

        # 反向重建分段
        segments = []
        current = n
        info1 =[]
        for j in range(group_count, 0, -1):
            prev = path[current][j]
            segments.append([prev,current])
            current = prev
        segments.reverse()


        all_wc = []
        all_cc = []
        for seg in segments:
            all_wc.append((zpow-right[seg[0]][seg[1]])**2)
            all_cc.append(color_cost[seg[0]][seg[1]])
        
        std_wc = np.std(all_wc)+ 1e-8
        std_cc = np.std(all_cc)+ 1e-8
        mean_wc_new = np.mean(all_wc)
        mean_cc_new = np.mean(all_cc)
        min_wc_new = min(all_wc)
        min_cc_new = min(all_cc)
        
        if (abs(min_wc_new - min_wc)/(min_wc+1e-8) < 1e-3) and (abs(min_cc_new - min_cc)/(min_cc+1e-8) < 1e-3):
            break
        min_wc = min_wc_new
        min_cc = min_cc_new
    # print("Time is ",TIME)        
    for seg in segments:
        if (green_count[seg[1]] - green_count[seg[0]]) > (red_count[seg[1]] - red_count[seg[0]]):
            info1.append(1)
        elif ((green_count[seg[1]] - green_count[seg[0]]) == (red_count[seg[1]] - red_count[seg[0]])):
            info1.append(random.randint(0, 1))
        else:
            info1.append(0)
        
    return info1,beta