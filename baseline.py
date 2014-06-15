#encoding=utf-8
import  argparse, dataset_walker, json, time, copy
from collections import defaultdict


def labels(user_act, mact) :
    #找到this的上下文
    # get context for "this" in inform(=dontcare)
    #找到affirm和negate的上下文
    # get context for affirm and negate
    #槽的名字是this
    this_slot = None
    
    #sys:Sorry would you like world or mediterranean food?  (sys-acts:select(food=world),select(food=mediterranean))
    #usr:(ASR:mediterranean food)    (SLU:inform(food=mediterranean))    (ORACLE:mediterranean food)  (Oracle-acts:inform(food=mediterranean))

    #sys:There are  restaurants serving indian in the cheap price range . What area would you like?  (sys-acts:request(slot=area),impl-conf(food=indian),impl-conf(pricerange=cheap))
    #usr:(ASR:any area)  (SLU:inform(this=dontcare),inform(area=dontcare))   (ORACLE:any area)  (Oracle-acts:inform(area=dontcare))

    #sys:Ok , a restaurant in any part of town is that right?  (sys-acts:expl-conf(area=dontcare))
    #usr:(ASR:there british food)    (SLU:inform(food=british))  (ORACLE:serves british food)  (Oracle-acts:inform(food=british))

    #消解this的指向
    confirm_slots = {"explicit":[], "implicit":[]}
    for act in mact :
        if act["act"] == "request" :
            this_slot = act["slots"][0][1]
        elif act["act"] == "select" :
            this_slot = act["slots"][0][0]
        elif act["act"] == "impl-conf":
            confirm_slots["implicit"] += act["slots"]
        elif act["act"] == "expl-conf" :
            confirm_slots["explicit"] += act["slots"]
            this_slot = act["slots"][0][0]
                    
            
            
    # goal_labels
    informed_goals = {}
    denied_goals = defaultdict(list)
    #user_act是同一轮对话轮，同一个对话轮的假设，这个假设中可能包含若干子句
    for act in user_act :
        act_slots = act["slots"]
        slot = None
        value = None
        #当前动作存在槽值对的情况下
        if len(act_slots) > 0:
            assert len(act_slots) == 1
            
            #第一个槽若是this，就把this_slot的值给slot
            if act_slots[0][0] == "this" :
                slot = this_slot
            else :
                slot = act_slots[0][0]
            value = act_slots[0][1]
        #经过以上操作slot，value存储的是act的一个槽值对        
        
        if act["act"] == "inform" and slot != None:
            informed_goals[slot]=(value)
            
        elif act["act"] == "deny" and slot != None:
            denied_goals[slot].append(value)
            
        elif act["act"] == "negate" :
            #negate否定的是机器给予的槽值对,本身没有槽值对
            #sys:Let me confirm , You are looking for a restaurant and you dont care about the price range right?  (sys-acts:expl-conf(pricerange=dontcare))
            #usr:(ASR:no)    (SLU:)  (ORACLE:no)  (Oracle-acts:negate())
            slot_values = confirm_slots["implicit"] + confirm_slots["explicit"]
            if len(slot_values) > 1:
                #否定多个槽值对，不清楚应该否定哪个槽值对
                #print "Warning: negating multiple slots- it's not clear what to do."
                pass
            else :
                for slot, value in slot_values :
                    denied_goals[slot].append(value)
            
        elif act["act"] == "affirm" :
            #affirm肯定的是机器给予的槽值对,本身没有槽值对
            #sys:Did you say you are looking for a restaurant in the south of town?  (sys-acts:expl-conf(area=south))
            #usr:(ASR:yes)   (SLU:affirm())  (ORACLE:yes)  (Oracle-acts:affirm())
            slot_values = confirm_slots["explicit"]
            if len(slot_values) > 1:
                #print "Warning: affirming multiple slots- it's not clear what to do."
                pass
            else :
                for slot, value in confirm_slots["explicit"] :
                    informed_goals[slot]=(value)
                    
    
          
    # requested slots
    #sys:peking restaurant is a nice place in the south of town serving tasty chinese food  (sys-acts:offer(name=peking restaurant),inform(food=chinese),inform(area=south))
    #usr:(ASR:what's their address)  (SLU:request(slot=addr))    (ORACLE:whats their address)  (Oracle-acts:request(addr))
    requested_slots = []
    for act in user_act :
        if act["act"] == "request" :
            for _, requested_slot in act["slots"]:
                requested_slots.append(requested_slot)
    # method
    method="none"
    act_types = [act["act"] for act in user_act]
    mact_types = [act["act"] for act in mact]
    
    if "reqalts" in act_types :
        method = "byalternatives"
    elif "bye" in act_types :
        method = "finished"
    elif "inform" in act_types:
        method = "byconstraints"
        for act in [uact for uact in user_act if uact["act"] == "inform"] :
            slots = [slot for slot, _ in act["slots"]]
            if "name" in slots :
                method = "byname"
            
    
            
    return informed_goals, denied_goals, requested_slots, method

  
#对语料中槽去this化
def Uacts(turn) :
    # return merged slu-hyps, replacing "this" with the correct slot
    mact = []

    #从turn中取出机器的动作
    if "dialog-acts" in turn["output"] :
        mact = turn["output"]["dialog-acts"]

    #this_slot是什么？
    this_slot = None

    #this_slot应该是机器的槽
    for act in mact :
        #act有两个键，一个act，一个slots
        if act["act"] == "request" :
            #取机器act的第一个槽的值给this_slot
            this_slot = act["slots"][0][1]

    #this_output是什么？
    this_output = []

    #turn['input']['live']['slu-hyps']是字典的列表，字典是以slu-hyp和score为键的字典，slu-hyp一个字典的列表，字典里面有键act和slots
    #下面这个循环遍历了所有可能的slu-hyp
    for slu_hyp in turn['input']["live"]['slu-hyps'] :
        score = slu_hyp['score']

        #this_slu_hyp 是个列表，每个列表项是一个字典，字典里面有动作和槽值对，包含了用户一个话轮中的所有分句,this_slu_hyp是用户一个话轮中所有（动作，槽值对）的列表，一个话轮可能有多个这样的列表级别的假设
        this_slu_hyp = slu_hyp['slu-hyp']

        #these_hyps
        these_hyps =  []

        #sys:What kind of food would you like?  (sys-acts:request(slot=food))
        #usr:(ASR:ok don't in)   (SLU:inform(this=dontcare)) (ORACLE:italian)  (Oracle-acts:inform(food=italian))

        #this_slu_hyp里面的子句的hyp经过把this处理掉装入these_hyps
        for  hyp in this_slu_hyp :
            for i in range(len(hyp["slots"])) :
                #只取出槽忽略值
                slot,_ = hyp["slots"][i]
                #(SLU:inform(this=dontcare))
                #这个例子里this变成了food，food=dontcare
                if slot == "this" :
                    hyp["slots"][i][0] = this_slot
            #these_hyps是这一话轮的所有（动作，槽值对）的hyp放在一个列表里面
            these_hyps.append(hyp)

        this_output.append((score, these_hyps))

    #对所有话轮的hyps假设按照score从大到小排序，用户每个话论有多个hyp，从子句的hyp看，对话状态表示成hyps是可以说的通的
    this_output.sort(key=lambda x:x[0], reverse=True)
    return this_output


#基线系统的状态跟踪器
class Tracker(object):
    def __init__(self):
        self.reset()
        
    #核心算法    
    #turn是个字典，键值有3种，output,turn-index,input
    def addTurn(self, turn):
        #把当前话轮之前的话轮形成的对话状态复制出来，以此为基础修改。
        hyps = copy.deepcopy(self.hyps)
        
        #从turn中取出系统的动作,dialog-acts包括系统的动作和槽值对
        #dialog-acts是个字典的数组,数组中的每个字典都是有一个系统动作和对应的槽值对
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        else :
            mact = []

        # clear requested-slots that have been informed
        #为什么要清空已经被informed的槽?这里的inform应该是由用户发起的?
        #这里的inform动作是机器发出的，机器看到用户request某些槽的值，就告知用户这些槽的相关信息,一旦用户被告知该槽的值，该槽就不再是requested的状态,所以需要对该requested-slots进行清零操作
        for act in mact :
            if act["act"] == "inform" :
                for slot,value in act["slots"]:
                    if slot in hyps["requested-slots"] :
                        hyps["requested-slots"][slot] = 0.0

        #Uacts,用户动作去this化，this=dontcare变成food=dontcare
        slu_hyps = Uacts(turn)
        
        #requested_slot_stats,method_stats,goal_stats
        requested_slot_stats = defaultdict(float)
        method_stats = defaultdict(float)
        goal_stats = defaultdict(lambda : defaultdict(float))
        
        #prev是先前，但不是先前对话轮，而是当前对话轮的上一个slu_hyp假设
        prev_method = "none"
        
        if len(hyps["method-label"].keys())> 0 :
            prev_method = hyps["method-label"].keys()[0]

        #对slu_hyps做循环，一个话轮的所有用户的假设都那过来推断对话状态,根本不是在NBest中去TopOne做
        for score, uact in slu_hyps :
            #uact是一个列表，列表里面是字典，字典的两个键分别是act和slots，一个字典代表一个子句
            #mact是机器的动作，和uact的结构一样
            informed_goals, denied_goals, requested, method = labels(uact, mact)

            # requested
            for slot in requested:
                requested_slot_stats[slot] += score

            #method
            if method == "none" :
                method = prev_method
            method_stats[method] += score

            # goal_labels
            for slot in informed_goals:
                value = informed_goals[slot]
                goal_stats[slot][value] += score
        
        # pick top values for each slot
        for slot in goal_stats:
            curr_score = 0.0
            if (slot in hyps["goal-labels"]) :
                #这个算法没有为slot存储多个值，而是之存储一个值
                curr_score = hyps["goal-labels"][slot].values()[0]
            for value in goal_stats[slot]:
                #goal_stats里面每个slot存储多个value
                score = goal_stats[slot][value]
                if score >= curr_score :
                    hyps["goal-labels"][slot] = {
                            value:clip(score)
                        }
                    curr_score = score
                    
        # joint estimate is the above selection, with geometric mean score
        goal_joint_label = {"slots":{}, "scores":[]}
        for slot in  hyps["goal-labels"] :
            (value,score), = hyps["goal-labels"][slot].items()
            #槽的值对应的置信度打分低于0.5不再计入goal_joint_label里面
            if score < 0.5 :
                # then None is more likely
                continue
            goal_joint_label["scores"].append(score)
            goal_joint_label["slots"][slot]= value
            
        #把goal_labels里面的槽值对及对应的置信度打分值对goal_joint_label生成联合概率
        if len(goal_joint_label["slots"]) > 0 :
            geom_mean = 1.0
            for score in goal_joint_label["scores"] :
                geom_mean *= score
            geom_mean = geom_mean**(1.0/len(goal_joint_label["scores"]))
            goal_joint_label["score"] = clip(geom_mean)
            del goal_joint_label["scores"]
            
            hyps["goal-labels-joint"] = [goal_joint_label]
        
        for slot in requested_slot_stats :
            hyps["requested-slots"][slot] = clip(requested_slot_stats[slot])
            
        # normalise method_stats    
        hyps["method-label"] = normalise_dict(method_stats)
        self.hyps = hyps 
        return self.hyps

    def reset(self):
        #HIS-POMDP的对话状态／对话假设是由（用户目的分区，用户动作，对话历史）三部分构成的，但是这里的对话状态和对话假设不是这样的，HIS的学习过程给我造成了一定的误导
        #self.hyps记录从对话开始到当前轮对于用户目的的跟踪。
        #这种想法是不对的，hyps之所以是复数是因为可以同时保存一个NBest列表的对话状态假设。
        #下面这些4项键值将会输入到最终的结果文件提交给主办方评测。
        #self.hyps是一个字典，hyp后面的s复数形式容易让人以为可以有多个对话状态假设候选，实际上并不是这样
        self.hyps = {"goal-labels":{}, "goal-labels-joint":[], "requested-slots":{}, "method-label":{}}
    


class FocusTracker(object):
    #下面这个注释是不对的，提交的结果里面需要有requested slots和method，不去跟踪这两个东西显然不对，Focus应该是相比于前面的对话历史跟注重当前话论提供的信息
    # only track goals, don't do requested slots and method
    def __init__(self):
        self.reset()
        
    #核心算法
    #turn是个字典，键值有3种，output,turn-index,input
    def addTurn(self, turn):
        hyps = copy.deepcopy(self.hyps)
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        else :
            mact = []
        slu_hyps = Uacts(turn)
       
        #this_u,method_stats,requested slots，对比前面的goal_stats这里没有
        this_u = defaultdict(lambda : defaultdict(float))
        method_stats = defaultdict(float)
        requested_slot_stats = defaultdict(float)

        #对slu_hyps做循环，一个话轮的所有用户的假设都那过来推断对话状态,根本不是在NBest中去TopOne做
        for score, uact in slu_hyps :
            #uact是一个列表，列表里面是字典，字典的两个键分别是act和slots，一个字典代表一个子句
            #mact是机器的动作，和uact的结构一样
            informed_goals, denied_goals, requested, method = labels(uact, mact)

            #method
            method_stats[method] += score

            #requested
            for slot in requested:
                requested_slot_stats[slot] += score

            # goal_labels
            for slot in informed_goals:
                this_u[slot][informed_goals[slot]] += score

        # in focus manner,前面的话轮的用户目的乘上一个q去打折                        
        for slot in this_u.keys() + hyps["goal-labels"].keys() :
            q = max(0.0,1.0-sum([this_u[slot][value] for value in this_u[slot]])) # clipping at zero because rounding errors
            if slot not in hyps["goal-labels"] :
                hyps["goal-labels"][slot] = {}
                
            for value in hyps["goal-labels"][slot] :
                
                hyps["goal-labels"][slot][value] *= q

            prev_values = hyps["goal-labels"][slot].keys()

            for value in this_u[slot] :
                if value in prev_values :
                    hyps["goal-labels"][slot][value] += this_u[slot][value]
                else :
                    hyps["goal-labels"][slot][value]=this_u[slot][value]
        
            hyps["goal-labels"][slot] = normalise_dict(hyps["goal-labels"][slot])
        
        # method node, in 'focus' manner:
        q = min(1.0,max(0.0,method_stats["none"]))
        method_label = hyps["method-label"]
        for method in method_label:
            if method != "none" :
                method_label[method] *= q
        for method in method_stats:
            if method == "none" :
                continue
            if method not in method_label :
                method_label[method] = 0.0
            method_label[method] += method_stats[method]
        
        if "none" not in method_label :
            method_label["none"] = max(0.0, 1.0-sum(method_label.values()))
        
        hyps["method-label"] = normalise_dict(method_label)
        
        # requested slots
        informed_slots = []
        for act in mact :
            if act["act"] == "inform" :
                for slot,value in act["slots"]:
                    informed_slots.append(slot)
                    
        for slot in (requested_slot_stats.keys() + hyps["requested-slots"].keys()):
            p = requested_slot_stats[slot]
            prev_p = 0.0
            if slot in hyps["requested-slots"] :
                prev_p = hyps["requested-slots"][slot]
            x = 1.0-float(slot in informed_slots)
            new_p = x*prev_p + p
            hyps["requested-slots"][slot] = clip(new_p)
            
        
            
        self.hyps = hyps 
        return self.hyps
    
    def reset(self):
        self.hyps = {"goal-labels":{},"method-label":{}, "requested-slots":{}}
    

def clip(x) :
    if x > 1:
        return 1
    if x<0 :
        return 0
    return x

#归一化字典里面所有的值
def normalise_dict(x) :
    x_items = x.items()
    total_p = sum([p for k,p in x_items])
    if total_p > 1.0 :
        x_items = [(k,p/total_p) for k,p in x_items]
    return dict(x_items)


def main() :
    
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH',
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE',
                        help='File to write with tracker output')
    parser.add_argument('--focus',dest='focus',action='store',nargs='?',default="False",const="True",
                        help='Use focus node tracker')
    args = parser.parse_args()
    #dataset文件中有多少对话，dataset就有多少对话，dataset是一个dataset_walker类的对象
    dataset = dataset_walker.dataset_walker(args.dataset, dataroot=args.dataroot)
    track_file = open(args.trackfile, "wb")
    track = {"sessions":[]}
    track["dataset"]  = args.dataset
    start_time = time.time()

    if args.focus.lower() == "true":
        tracker = FocusTracker()
    elif args.focus.lower() == "false":
        tracker = Tracker()
    else:
        raise RuntimeError,'Dont recognize focus=%s (must be True or False)' % (args.focus)    
    for call in dataset :
        #把dataset中的对话一个个拿出来按照话轮处理
        this_session = {"session-id":call.log["session-id"], "turns":[]}

        #tracker的reset操作:self.hyps = {"goal-labels":{},"method-label":{}, "requested-slots":{}}
        tracker.reset()

        #对每一个对话按照其中的话轮开始跟中用户的对话目的
        #跟踪对话状态:dialog state 或者说是 dialog hypthesis
        for turn, _ in call :
            #所以关于对于对话状态跟踪都在addTurn函数里面处理
            #最核心的代码，核心算法部分。
            #turn是个字典，键值有3种，output,turn-index,input
            ####################################
            tracker_turn = tracker.addTurn(turn)
            ####################################
            this_session["turns"].append(tracker_turn)
        
        #把当前对话所有话轮的用户目的的跟中结果放在this_session，添加到最终的结果集合track中。
        track["sessions"].append(this_session)
    end_time = time.time()
    elapsed_time = end_time - start_time
    track["wall-time"] = elapsed_time
   
    json.dump(track, track_file,indent=4)

import sys
    
if __name__ == '__main__':
    main()
