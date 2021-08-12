from mlrunner.utils import *
# include whatever other imports are needed (e.g. the ml algorithm, environment)
import multiprocessing as mp
from tpg.trainer import Trainer, loadTrainer
from tpg.utils import actionInstructionStats, getTeams, getLearners
import gym
import time
from math import sin, pi, cos
from numpy import append, mean, std

"""
Creates a new trainer for this run.
"""
def create_trainer(environment, init_team_pop, gap, registers,
        init_max_team_size, max_team_size, init_max_prog_size,
        learner_del_prob, learner_add_prob, learner_mut_prob, prog_mut_prob,
        act_mut_prob, atomic_act_prob, inst_del_prob,
        inst_add_prob, inst_swp_prob, inst_mut_prob, elitist, rampancy,
        ops, init_max_act_prog_size):
    # get info about the environment from temp env
    env = gym.make(environment)
    acts = env.action_space.shape[0]
    inpts = env.observation_space.shape[0] + 6 # plus 6 for sin+cos inputs
    del env

    return Trainer(actions=[acts,acts], 
        teamPopSize=init_team_pop, gap=gap, nRegisters=registers,
        initMaxTeamSize=init_max_team_size, maxTeamSize=max_team_size,
        initMaxProgSize=init_max_prog_size, pLrnDel=learner_del_prob,
        pLrnAdd=learner_add_prob, pLrnMut=learner_mut_prob,
        pProgMut=prog_mut_prob, pActMut=act_mut_prob,
        pActAtom=atomic_act_prob,
        inputSize=inpts,
        pInstDel=inst_del_prob, pInstAdd=inst_add_prob,
        pInstMut=inst_swp_prob, pInstSwp=inst_mut_prob,
        doElites=elitist, rampancy=rampancy,
        operationSet=ops, initMaxActProgSize=init_max_act_prog_size,
        nActRegisters=acts+4)

def run_agent(args):
    agent = args[0] # the agent
    env_name = args[1] # name of gym environment
    score_list = args[2] # track scores of all agents
    episodes = args[3] # number of times to repeat game
    frames = args[4] # frames to play for
    agent_num = args[5] # index to store results in list

    agent.configFunctionsSelf()
    scores = []

    for ep in range(episodes): # episode loop

        # create new env each time (restart doesn't work well)
        env = gym.make(env_name)
        state = env.reset()
        score_ep = 0
        agent.zeroRegisters()

        for i in range(frames): # frame loop

            state = append(state, [2*sin(0.2*pi*i), 2*cos(0.2*pi*i),
                               2*sin(0.1*pi*i), 2*cos(0.1*pi*i),
                               2*sin(0.05*pi*i), 2*cos(0.05*pi*i)])

            act = agent.act(state)[1]

            # feedback from env
            state, reward, is_done, _ = env.step(act)

            score_ep += reward # accumulate reward in score
            if is_done:
                break # end early if losing state

        scores.append(score_ep)

        env.close()
        del env

    final_score = mean(scores)
    print(f"agent {str(agent.agentNum)} scored: {str(final_score)}")

    agent.reward(final_score, env_name)
    agent.reward(scores, "Scores")
    agent.reward(final_score, "Mean")
    agent.reward(std(scores), "Std")

    score_list[agent_num] = (agent.team.id, agent.team.outcomes)


"""
Runs the experiment from start to finish.
"""
def run_experiment(instance=0, end_generation=500, episodes=5, 
        environment="BipedalWalker-v3", frames=1600, processes=1,
        trainer_checkpoint=100, init_team_pop=360, gap=0.5, registers=8,
        init_max_team_size=2, max_team_size=4, init_max_prog_size=128,
        learner_del_prob=0.3, learner_add_prob=0.3, learner_mut_prob=0.7,
        prog_mut_prob=0.5, act_mut_prob=0.7, atomic_act_prob=0.8,
        init_max_act_prog_size=128, inst_del_prob=0.5, inst_add_prob=0.5,
        inst_swp_prob=0.5, inst_mut_prob=0.5, elitist=True, ops="robo",
        rampancy=(5,5,5), hh_remove_gen=100, fail_gens=100):

    if __name__ == "__main__":
        mp.set_start_method("spawn")

    # log the experiment parameters
    with open(f"params-run{instance}.txt", "w") as f:
        f.write(f"end_generation: {str(end_generation)}\n")
        f.write(f"episodes: {str(episodes)}\n")
        f.write(f"environment: {str(environment)}\n")
        f.write(f"frames: {str(frames)}\n")
        f.write(f"processes: {str(processes)}\n")
        f.write(f"trainer_checkpoint: {str(trainer_checkpoint)}\n")
        f.write(f"init_team_pop: {str(init_team_pop)}\n")
        f.write(f"gap: {str(gap)}\n")
        f.write(f"registers: {str()}\n")
        f.write(f"init_max_team_size: {str()}\n")
        f.write(f"max_team_size: {str()}\n")
        f.write(f"init_max_prog_size: {str()}\n")
        f.write(f"learner_del_prob: {str()}\n")
        f.write(f"learner_add_prob: {str()}\n")
        f.write(f"learner_mut_prob: {str()}\n")
        f.write(f"prog_mut_prob: {str()}\n")
        f.write(f"act_mut_prob: {str()}\n")
        f.write(f"atomic_act_prob: {str()}\n")
        f.write(f"init_max_act_prog_size: {str(init_max_act_prog_size)}\n")
        f.write(f"inst_del_prob: {str(inst_del_prob)}\n")
        f.write(f"inst_add_prob: {str(inst_add_prob)}\n")
        f.write(f"inst_swp_prob: {str(inst_swp_prob)}\n")
        f.write(f"inst_mut_prob: {str(inst_mut_prob)}\n")
        f.write(f"elitist: {str(elitist)}\n")
        f.write(f"ops: {str(ops)}\n")
        f.write(f"rampancy: {str(rampancy)}\n")
        f.write(f"hh_remove_gen: {str()}\n")
        f.write(f"fail_gens: {str()}\n")

    # continue old run if applicable
    try:
        trainer = loadTrainer(f"trainer-run{instance}.pkl")
    except:
        trainer = None

    if trainer is not None:
        # start from previous point
        start_gen = trainer.generation
    else:
        # create new trainer and log, start at gen 0
        trainer = create_trainer(environment, init_team_pop, gap, registers,
        init_max_team_size, max_team_size, init_max_prog_size,
        learner_del_prob, learner_add_prob, learner_mut_prob, prog_mut_prob,
        act_mut_prob, atomic_act_prob, inst_del_prob,
        inst_add_prob, inst_swp_prob, inst_mut_prob, elitist, rampancy,
        ops, init_max_act_prog_size)

        start_gen = 0
        create_log(f"log-run{instance}.csv",[
            "gen", "gen-time", "fit-min", "fit-max", "fit-avg", "gen-max",
            "champ-id", "champ-mean", "champ-std", "champ-teams",
            "champ-learners", "champ-inst", "champ-act-inst", "champ-real-acts",
            "pop-roots", "pop-teams", "pop-learners", "hh-learners-rem",
            "hh-teams-affected", "b-teams", "b-learners", 
            "b-teams-new", "b-learners-new"
        ])

    # set up multiprocessing
    man = mp.Manager()
    pool = mp.Pool(processes=processes, maxtasksperchild=1)

    # where the run actually happens
    for gen in range(start_gen, end_generation):

        # get all agents to execute for current generation
        agents = trainer.getAgents(skipTasks=[environment])

        # multiprocess list to store results
        score_list = man.list(range(len(agents)))

        gen_start_time = time.time()

        pool.map(run_agent,
            [(agent, environment, score_list, episodes, 
                    frames, i)
                for i, agent in enumerate(agents)])

        gen_time = int(time.time() - gen_start_time)

        # manually apply these scores to  teams
        trainer.applyScores(score_list)

        # save trainer at this point
        if gen % trainer_checkpoint == 0:
            trainer.saveToFile(f"trainer-run{instance}-{gen}.pkl")

        # get basic generational stats
        champ_agent = trainer.getEliteAgent(environment)
        insts = actionInstructionStats([champ_agent.team.learners[0]], 
                                    trainer.operations)["overall"]["total"]

        # evolve and save trainer backup / final trainer
        trainer.evolve(tasks=[environment])
        trainer.saveToFile(f"trainer-run{instance}.pkl")

        # find max fitness obtained this generation
        max_this_gen = -99999
        for s in score_list:
            if s[1]["Mean"] > max_this_gen:
                max_this_gen = s[1]["Mean"]

        # log the current generation data
        update_log(f"log-run{instance}.csv",[
            gen, gen_time, round(trainer.fitnessStats["min"], 4), 
            round(trainer.fitnessStats["max"], 4), 
            round(trainer.fitnessStats["average"], 4), round(max_this_gen, 4),
            champ_agent.team.id, round(champ_agent.team.outcomes["Mean"], 4),
            round(champ_agent.team.outcomes["Std"], 4), insts
        ])