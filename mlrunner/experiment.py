from utils import *
# include whatever other imports are needed (e.g. the ml algorithm, environment)
import multiprocessing as mp
from optparse import OptionParser
from tpg.trainer import Trainer, loadTrainer
from tpg.utils import actionInstructionStats, getTeams, getLearners
import gym
import time
from math import sin, pi, cos
from numpy import append, mean, std, clip

"""
Utility function to parse int array input parameters.
"""
def separgs_int(option, opt, value, parser):
    ints = []
    for value in value.split(','):
        ints.append(int(value))

    setattr(parser.values, option.dest, ints)

parser = OptionParser()

# parameters for running experiment
parser.add_option("--end_generation", type="int", dest="end_generation", default=500)
parser.add_option("--episodes", type="int", dest="episodes", default=5)
parser.add_option("--environment", type="string", dest="environment")
parser.add_option("--frames", type="int", dest="frames", default=1600)
parser.add_option("--processes", type="int", dest="processes", default=1)
parser.add_option("--trainer_checkpoint", type="int", dest="trainer_checkpoint", default=100)

# parameters for tpg (gp in this experiment)
parser.add_option("--init_team_pop", type="int", dest="init_team_pop", default=360)
parser.add_option("--gap", type="float", dest="gap", default=0.5)
parser.add_option("--init_max_act_prog_size", type="int", dest="init_max_act_prog_size", default=128)
parser.add_option("--inst_del_prob", type="float", dest="inst_del_prob", default=0.5)
parser.add_option("--inst_add_prob", type="float", dest="inst_add_prob", default=0.5)
parser.add_option("--inst_swp_prob", type="float", dest="inst_swp_prob", default=0.5)
parser.add_option("--inst_mut_prob", type="float", dest="inst_mut_prob", default=0.5)
parser.add_option("--elitist", action="store_true", dest="elitist", default=True)
parser.add_option("--ops", type="string", dest="ops", default="robo")
parser.add_option("--rampancy", type="string", action="callback", callback=separgs_int, dest="rampancy", default="5,5,5")

(opts, args) = parser.parse_args()

"""
Creates a new trainer for this run.
"""
def create_trainer():
    # get info about the environment from temp env
    env = gym.make(opts.environment)
    acts = env.action_space.shape[0]
    inpts = env.observation_space.shape[0] + 6 # plus 6 for sin+cos inputs
    del env

    return Trainer(actions=[acts,acts], 
        teamPopSize=opts.init_team_pop, gap=opts.gap, 
        inputSize=inpts,
        pInstDel=opts.inst_del_prob, pInstAdd=opts.inst_add_prob,
        pInstMut=opts.inst_swp_prob, pInstSwp=opts.inst_mut_prob,
        doElites=opts.elitist, rampancy=opts.rampancy,
        operationSet=opts.ops, initMaxActProgSize=opts.init_max_act_prog_size,
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
def run_experiment():
    run_name = f"{opts.run_key}-{opts.environment}-{opts.instance}"
    log_name = "log.txt"

    # continue old run if applicable
    try:
        trainer = loadTrainer("trainer.pkl")
    except:
        trainer = None

    if trainer is not None:
        # start from previous point
        start_gen = trainer.generation
    else:
        # create new trainer and log, start at gen 0
        trainer = create_trainer()
        start_gen = 0
        create_log("log.csv",[
            "gen", "gen-time", "fit-min", "fit-max", "fit-avg", "champ-id",
            "champ-mean", "champ-std", "champ-act-inst"
        ])

    # set up multiprocessing
    man = mp.Manager()
    pool = mp.Pool(processes=opts.processes, maxtasksperchild=1)

    # where the run actually happens
    for gen in range(start_gen, opts.end_generation):

        # get all agents to execute for current generation
        agents = trainer.getAgents(skipTasks=[opts.environment])

        # multiprocess list to store results
        score_list = man.list(range(len(agents)))

        gen_start_time = time.time()

        pool.map(run_agent,
            [(agent, opts.environment, score_list, opts.episodes, 
                    opts.frames, i)
                for i, agent in enumerate(agents)])

        gen_time = int(time.time() - gen_start_time)

        # manually apply these scores to  teams
        trainer.applyScores(score_list)

        # save trainer at this point
        if gen % opts.trainer_checkpoint == 0:
            trainer.saveToFile(f"trainer-{gen}.pkl")

        # get basic generational stats
        champ_agent = trainer.getEliteAgent(opts.environment)
        insts = actionInstructionStats([champ_agent.team.learners[0]], 
                                    trainer.operations)["overall"]["total"]

        # evolve and save trainer backup / final trainer
        trainer.evolve(tasks=[opts.environment])
        trainer.saveToFile(f"trainer.pkl")

        # log the current generation data
        update_log("log.csv",[
            gen, gen_time, round(trainer.fitnessStats["min"], 4), 
            round(trainer.fitnessStats["max"], 4), 
            round(trainer.fitnessStats["average"], 4), 
            champ_agent.team.id, round(champ_agent.team.outcomes["Mean"], 4),
            round(champ_agent.team.outcomes["Std"], 4), insts
        ])

# start the experiment
if __name__ == "__main__":
    mp.set_start_method("spawn")
    run_experiment()