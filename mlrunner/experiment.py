from mlrunner.utils import *
# include whatever other imports are needed (e.g. the ml algorithm, environment)
import multiprocessing as mp
from tpg.trainer import Trainer, loadTrainer
from tpg.utils import learnerInstructionStats, actionInstructionStats, getTeams, getLearners
import aicrowd_gym
import nle
import gym
import time
from numpy import append, mean, std, array

"""
Convert the dictionary observations to a single numeric list.
"""
def get_linear_obs(obs):
    new_obs = []

    new_obs.extend(obs["glyphs"].flatten())
    new_obs.extend(obs["chars"].flatten())
    new_obs.extend(obs["colors"].flatten())
    new_obs.extend(obs["specials"].flatten())
    new_obs.extend(obs["blstats"])
    new_obs.extend(obs["message"])
    new_obs.extend(obs["inv_glyphs"])
    new_obs.extend(obs["inv_strs"].flatten())
    new_obs.extend(obs["inv_letters"])
    new_obs.extend(obs["inv_oclasses"])
    new_obs.extend(obs["tty_chars"].flatten())
    new_obs.extend(obs["tty_colors"].flatten())
    new_obs.extend(obs["tty_cursor"])
    new_obs.extend(obs["misc"])

    return new_obs


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
    env = aicrowd_gym.make(environment)
    state = env.reset()
    inpts = len(get_linear_obs(state))
    del env

    return Trainer(actions=[7,7], 
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
        nActRegisters=registers)

def run_agent(args):
    agent = args[0] # the agent
    env_name = args[1] # name of gym environment
    episodes = args[2] # number of times to repeat game
    frames = args[3] # frames to play for

    agent.configFunctionsSelf()
    scores = []

    for ep in range(episodes): # episode loop

        # create new env each time (restart doesn't work well)
        env = gym.make(env_name)
        state = env.reset()
        score_ep = 0
        agent.zeroRegisters()

        for i in range(frames): # frame loop

            state = get_linear_obs(state)

            # get action from binary register values.
            act0 = array(agent.act(state)[1]) >= 0.5
            act = 64*act0[0] + 32*act0[1] + 16*act0[2] + 8*act0[3] + 4*act0[4] + 2*act0[5] + act0[6]

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

    return (agent.team.id, agent.team.outcomes)


"""
Runs the experiment from start to finish.
"""
def run_experiment(instance=0, end_generation=10000, episodes=1, 
        environment="NetHackChallenge-v0", frames=99999, processes=1,
        trainer_checkpoint=1000, init_team_pop=360, gap=0.5, registers=10,
        init_max_team_size=2, max_team_size=5, init_max_prog_size=256,
        learner_del_prob=0.3, learner_add_prob=0.3, learner_mut_prob=0.7,
        prog_mut_prob=0.5, act_mut_prob=0.7, atomic_act_prob=0.7,
        init_max_act_prog_size=256, inst_del_prob=0.5, inst_add_prob=0.5,
        inst_swp_prob=0.5, inst_mut_prob=0.5, elitist=True, ops="robo",
        rampancy=(5,5,5), hh_remove_gen=100, fail_gens=100, sbb_gens=1000,
        partial_start=False, sbb_n=20):

    mp.set_start_method("spawn")
    #print(1/0)

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
        f.write(f"registers: {str(registers)}\n")
        f.write(f"init_max_team_size: {str(init_max_team_size)}\n")
        f.write(f"max_team_size: {str(max_team_size)}\n")
        f.write(f"init_max_prog_size: {str(init_max_prog_size)}\n")
        f.write(f"learner_del_prob: {str(learner_del_prob)}\n")
        f.write(f"learner_add_prob: {str(learner_add_prob)}\n")
        f.write(f"learner_mut_prob: {str(learner_mut_prob)}\n")
        f.write(f"prog_mut_prob: {str(prog_mut_prob)}\n")
        f.write(f"act_mut_prob: {str(act_mut_prob)}\n")
        f.write(f"atomic_act_prob: {str(atomic_act_prob)}\n")
        f.write(f"init_max_act_prog_size: {str(init_max_act_prog_size)}\n")
        f.write(f"inst_del_prob: {str(inst_del_prob)}\n")
        f.write(f"inst_add_prob: {str(inst_add_prob)}\n")
        f.write(f"inst_swp_prob: {str(inst_swp_prob)}\n")
        f.write(f"inst_mut_prob: {str(inst_mut_prob)}\n")
        f.write(f"elitist: {str(elitist)}\n")
        f.write(f"ops: {str(ops)}\n")
        f.write(f"rampancy: {str(rampancy)}\n")
        f.write(f"hh_remove_gen: {str(hh_remove_gen)}\n")
        f.write(f"fail_gens: {str(fail_gens)}\n")
        f.write(f"sbb_gens: {str(sbb_gens)}\n")
        f.write(f"partial_start: {str(partial_start)}\n")
        f.write(f"sbb_n: {str(sbb_n)}\n")

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

        trainer.no_improvements = 0
        trainer.cur_max = -99999
        trainer.last_b_gen = 0
        trainer.total_gens = 0
        trainer.gen_b = 0
        trainer.old_max = -99999

        start_gen = 0
        create_log(f"log-run{instance}.csv",[
            "gen", "tpg-gen", "gen-time", "fit-min", "fit-max", "fit-avg", "gen-max",
            "champ-id", "champ-mean", "champ-std", "champ-teams",
            "champ-learners", "champ-bid-inst", "champ-act-inst", "champ-real-acts",
            "pop-roots", "pop-teams", "pop-learners", "hh-learners-rem",
            "hh-teams-affected", "b-teams", "b-learners", 
            "b-teams-new", "b-learners-new"
        ])

        create_log(f"log-b-run{instance}.csv",[
            "gen", "tpg-gen", "sbb-gen", "gen-time", "fit-min", "fit-max", "fit-avg", "gen-max",
            "champ-id", "champ-mean", "champ-std", "champ-teams", 
            "champ-learners", "champ-bid-inst", 
            "champ-act-inst", "champ-real-acts",
            "pop-roots", "pop-teams", "pop-learners"
        ])

    # set up multiprocessing
    pool = mp.Pool(processes=processes, maxtasksperchild=1)

    trainer.b_teams = None

    # where the run actually happens
    for gen in range(start_gen, end_generation):

        """
        Run SBB/B population here.
        """
        if (trainer.no_improvements >= fail_gens or gen == 0) and sbb_gens > 0:
            trainer.last_b_gen = gen

            # check for existing run
            if trainer.gen_b > 0:
                # load existing
                trainer_b = loadTrainer(f"trainer-b-run{instance}.pkl")
                trainer_b.configFunctions()
            else:
                # start new
                trainer_b = create_trainer(environment, init_team_pop, gap, registers,
                    init_max_team_size, max_team_size, init_max_prog_size,
                    learner_del_prob, learner_add_prob, learner_mut_prob, prog_mut_prob,
                    act_mut_prob, 1.0, inst_del_prob,
                    inst_add_prob, inst_swp_prob, inst_mut_prob, elitist, rampancy,
                    ops, init_max_act_prog_size)

                trainer_b.no_improvements = 0
                trainer_b.old_max = -99999


            # evolve the sbb/b trainer
            for gen_b in range(trainer_b.generation, sbb_gens):
                print(f"On gen {gen}, SBB gen {gen_b}.")

                agents = trainer_b.getAgents(skipTasks=[environment])
                gen_start_time = time.time()
                score_list = pool.map(run_agent,
                    [(agent, environment, episodes, frames)
                        for agent in agents])
                gen_time = int(time.time() - gen_start_time)
                trainer_b.applyScores(score_list)

                max_score = trainer_b.getEliteAgent(task=environment).team.outcomes[environment]

                trainer_b.evolve(tasks=[environment])

                # check if improvements
                if max_score > trainer_b.old_max:
                    trainer_b.old_max = max_score
                    trainer_b.no_improvements = 0
                else:
                    trainer_b.no_improvements += 1
                    if trainer_b.no_improvements > fail_gens:
                        break

                # get basic generational stats
                champ_agent = trainer_b.getEliteAgent(environment)
                champ_teams = getTeams(champ_agent.team)
                champ_learners = getLearners(champ_agent.team)
                n_champ_bid_insts = learnerInstructionStats(champ_learners, 
                                            trainer_b.operations)["overall"]["total"]
                n_champ_act_insts = actionInstructionStats(champ_learners, 
                                            trainer_b.operations)["overall"]["total"]
                n_champ_real_acts = len([l for l in champ_learners if l.isActionAtomic()])
                n_pop_roots = len(trainer_b.rootTeams)
                n_pop_teams = len(trainer_b.teams)
                n_pop_learners = len(trainer_b.learners)

                # find max fitness obtained this generation
                max_this_gen = -99999
                for s in score_list:
                    if s[1]["Mean"] > max_this_gen:
                        max_this_gen = s[1]["Mean"]

                # log the current generation data
                update_log(f"log-b-run{instance}.csv",[
                    trainer.total_gens, gen, gen_b, gen_time, round(trainer_b.fitnessStats["min"], 4), 
                    round(trainer_b.fitnessStats["max"], 4), 
                    round(trainer_b.fitnessStats["average"], 4), round(max_this_gen, 4),
                    champ_agent.team.id, round(champ_agent.team.outcomes["Mean"], 4),
                    round(champ_agent.team.outcomes["Std"], 4), len(champ_teams), 
                    len(champ_learners), n_champ_bid_insts, n_champ_act_insts, n_champ_real_acts,
                    n_pop_roots, n_pop_teams, n_pop_learners
                ])

                # save every gen
                trainer_b.saveToFile(f"trainer-b-run{instance}.pkl")
                trainer.gen_b = trainer_b.generation
                trainer.saveToFile(f"trainer-run{instance}.pkl")

                # break from total SBB and TPG generations
                trainer.total_gens += 1
                if trainer.total_gens == end_generation:
                    return

            # mark pop b individuals
            for i in range(len(trainer_b.teams)):
                trainer_b.teams[i].id2 = f"b_t_{trainer.generation}_{trainer_b.generation}"
            for i in range(len(trainer_b.learners)):
                trainer_b.learners[i].id2 = f"b_l_{trainer.generation}_{trainer_b.generation}"

            # save top few teams per start state
            trainer.b_teams = []
            top20 = trainer_b.getAgents(sortTasks=[environment])[:sbb_n]
            for agent in top20:
                if agent.team not in trainer.b_teams:
                    trainer.b_teams.append(agent.team)
            
            # reset no_improvements tracker so no immedate pop b / SBB repeat
            trainer.no_improvements = 0

        # reset tracker for generation of pop b
        trainer.gen_b = 0
        trainer.configFunctions()

        """
        Regular TPG evolution.
        """

        print(f"On gen {gen}.")

        # get all agents to execute for current generation
        agents = trainer.getAgents(skipTasks=[environment])

        gen_start_time = time.time()

        score_list = pool.map(run_agent,
            [(agent, environment, episodes, frames)
                for agent in agents])

        gen_time = int(time.time() - gen_start_time)

        # manually apply these scores to  teams
        trainer.applyScores(score_list)

        # save trainer at this point
        if gen % trainer_checkpoint == 0:
            trainer.saveToFile(f"trainer-run{instance}-{gen}.pkl")

        # get basic generational stats
        champ_agent = trainer.getEliteAgent(environment)
        champ_teams = getTeams(champ_agent.team)
        champ_learners = getLearners(champ_agent.team)
        n_champ_bid_insts = learnerInstructionStats(champ_learners, 
                                    trainer.operations)["overall"]["total"]
        n_champ_act_insts = actionInstructionStats(champ_learners, 
                                    trainer.operations)["overall"]["total"]
        n_champ_real_acts = len([l for l in champ_learners if l.isActionAtomic()])
        n_pop_roots = len(trainer.rootTeams)
        n_pop_teams = len(trainer.teams)
        n_pop_learners = len(trainer.learners)

        # find all from b pop
        n_b_teams = 0
        n_b_learners = 0
        n_b_teams_new = 0
        n_b_learners_new = 0
        for t in trainer.teams:
            try:
                tmp = t.id2
                n_b_teams += 1
                if int(tmp.split("_")[2]) == trainer.last_b_gen:
                    n_b_teams_new += 1
            except:
                pass

        for l in trainer.learners:
            try:
                tmp = l.id2
                n_b_learners += 1
                if int(tmp.split("_")[2]) == trainer.last_b_gen:
                    n_b_learners_new += 1
            except:
                pass

        # do hitchiker removal when past gen 0 on certain generations
        if gen % hh_remove_gen == 0 and gen > 0:

            # obtain team learner data
            teams, learners_visited = get_team_learner_visits(trainer.getAgents(), 
                environment, frames, trainer.teams, trainer.learners)

            # remove hitchhikers
            learners_removed_tmp, teams_affected_tmp = trainer.removeHitchhikers(
                teams, learners_visited)
            learners_removed = len(learners_removed_tmp)
            teams_affected = len(teams_affected_tmp)

        else:
            learners_removed = -1
            teams_affected = -1

        

        # find max fitness obtained this generation
        max_this_gen = -99999
        for s in score_list:
            if s[1]["Mean"] > max_this_gen:
                max_this_gen = s[1]["Mean"]

        # check if improvements
        if max_this_gen > trainer.old_max:
            trainer.old_max = max_this_gen
            trainer.no_improvements = 0
        else:
            trainer.no_improvements += 1

        # evolve and save trainer backup / final trainer
        trainer.evolve(tasks=[environment], extraTeams=trainer.b_teams)
        trainer.saveToFile(f"trainer-run{instance}.pkl")

        # log the current generation data
        update_log(f"log-run{instance}.csv",[
            trainer.total_gens, gen, gen_time, round(trainer.fitnessStats["min"], 4), 
            round(trainer.fitnessStats["max"], 4), 
            round(trainer.fitnessStats["average"], 4), round(max_this_gen, 4),
            champ_agent.team.id, round(champ_agent.team.outcomes["Mean"], 4),
            round(champ_agent.team.outcomes["Std"], 4), len(champ_teams), 
            len(champ_learners), n_champ_bid_insts, n_champ_act_insts, n_champ_real_acts,
            n_pop_roots, n_pop_teams, n_pop_learners, learners_removed, 
            teams_affected, n_b_teams, n_b_learners, n_b_teams_new, n_b_learners_new
        ])

        # break from total SBB and TPG generations
        trainer.total_gens += 1
        if trainer.total_gens == end_generation:
            break

'''
Traces path taken by all agents to find team learner visits, which is returned.
'''
def get_team_learner_visits(agents, env_name, frames, teams, learners):

    team_learner_visits_ids = {}

    print("Path tracing for team learner visits.")

    agent_i = 0

    for agent in agents:

        env = gym.make(env_name)

        print(f"Tracing agent # {agent_i+1} out of {len(agents)}")
        agent_i += 1

        agent.configFunctionsSelf()
        agent.zeroRegisters()

        state = env.reset()
        score = 0

        for i in range(frames):

            state = append(state, [2*sin(0.2*pi*i), 2*cos(0.2*pi*i),
                                2*sin(0.1*pi*i), 2*cos(0.1*pi*i),
                                2*sin(0.05*pi*i), 2*cos(0.05*pi*i)])

            trace = {}
        
            act = agent.act(state, path_trace=trace)[1]

            # feedback from env
            state, _, is_done, _ = env.step(act)

            # save learners visited per team
            for path_seg in trace["path"]:
                # create new list if team not visited yet
                if path_seg["team_id"] not in team_learner_visits_ids:
                    team_learner_visits_ids[path_seg["team_id"]] = []
                # add new learners to list of learners this team visited
                if path_seg["top_learner"] not in team_learner_visits_ids[path_seg["team_id"]]:
                    team_learner_visits_ids[path_seg["team_id"]].append(path_seg["top_learner"])

            if is_done:
                break # end early if losing state

        env.close()
        del env

    return get_team_learner_visits_objects(team_learner_visits_ids, teams, learners)

"""
Recreate team learner visits structure but with objects instead of ids.
"""
def get_team_learner_visits_objects(team_learner_visits_ids, teams, learners):

    team_learner_visits = {}
    teams_final = []
    learners_visited = []

    for team_id in team_learner_visits_ids.keys():
        print("checking...")
        #[print(t.id, "==", team_id, "->", str(t.id)==str(team_id)) for t in teams]
        team = [t for t in teams if str(t.id) == str(team_id)][0]

        # add the team into the dict
        teams_final.append(team)
        learners_visited.append([])

        # add team's visited learners to dict for that team
        for learner_id in team_learner_visits_ids[team_id]:
            learner = [l for l in learners if str(l.id) == str(learner_id)][0]

            learners_visited[-1].append(learner)

    return teams_final, learners_visited
