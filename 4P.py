import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive
import time

def main():
    fitness_rhc = []
    fitness_sa = []
    fitness_ga = []
    fitness_mmc = []
    
    time_rhc_list = []
    time_sa_list = []
    time_ga_list = []
    time_mmc_list = []
    #test different sizes of state space
    range_values = [5,10,20,50,100]
    for space_size in range_values:
        fitness = mlrose_hiive.FourPeaks(t_pct=0.1)
        problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2,size = space_size)
        start = time.time()
        _, opt_fitness_rhc, _ = mlrose_hiive.random_hill_climb(problem, max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)
        end = time.time()
        time_rhc = end-start
        
        start = time.time()
        _, opt_fitness_sa, _ = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)
        end = time.time()
        time_sa = end-start
        
        start = time.time()
        _, opt_fitness_ga, _ = mlrose_hiive.genetic_alg(problem, max_attempts = 1000, max_iters = 10000, curve = True)
        end = time.time()
        time_ga = end-start
        
        start = time.time()
        _, opt_fitness_mmc, _ = mlrose_hiive.mimic(problem, pop_size = 500, max_attempts = 100, max_iters = 10000, curve = True)
        end = time.time()
        time_mmc = end-start
        
        fitness_rhc.append(opt_fitness_rhc)
        fitness_sa.append(opt_fitness_sa)
        fitness_ga.append(opt_fitness_ga)
        fitness_mmc.append(opt_fitness_mmc)
        
        time_rhc_list.append(time_rhc)
        time_sa_list.append(time_sa)
        time_ga_list.append(time_ga)
        time_mmc_list.append(time_mmc)
        
    fitness_rhc = np.array(fitness_rhc)
    fitness_sa = np.array(fitness_sa)
    fitness_ga = np.array(fitness_ga)
    fitness_mmc = np.array(fitness_mmc)
    time_rhc_list = np.array(time_rhc_list)
    time_sa_list = np.array(time_sa_list)
    time_ga_list = np.array(time_ga_list)
    time_mmc_list = np.array(time_mmc_list)
	
    plt.figure()
    plt.plot(range_values, fitness_rhc, label = 'Randomized Hill Climbing')
    plt.plot(range_values, fitness_sa, label = 'Simulated Annealing')
    plt.plot(range_values, fitness_ga, label = 'Genetic Algorithm')
    plt.plot(range_values, fitness_mmc, label = 'MIMIC')
    plt.title('Fitness vs. State Space Size (Four Peaks)')
    plt.xlabel('Size')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('4P_fitness_vs_size.png')

    plt.figure()
    plt.plot(range_values, time_rhc_list, label = 'Randomized Hill Climbing')
    plt.plot(range_values, time_sa_list, label = 'Simulated Annealing')
    plt.plot(range_values, time_ga_list, label = 'Genetic Algorithm')
    plt.plot(range_values, time_mmc_list, label = 'MIMIC')
    plt.title('Time vs. State Space Size (Four Peaks)')
    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.legend()
    plt.savefig('4P_time_vs_size.png')

    #test different thresholds
    space_size = 50
    fitness = mlrose_hiive.FourPeaks(t_pct = 0.1)
    problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size = space_size)
    _, _, fitness_curve_rhc_1 = mlrose_hiive.random_hill_climb(problem, max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)
    _, _, fitness_curve_sa_1 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)    
    _, _, fitness_curve_ga_1 = mlrose_hiive.genetic_alg(problem, max_attempts = 1000, max_iters = 10000, curve = True)
    _, _, fitness_curve_mimic_1 = mlrose_hiive.mimic(problem, pop_size = 500, max_attempts = 100, max_iters = 10000, curve = True)

    fitness = mlrose_hiive.FourPeaks(t_pct = 0.2)
    problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size = space_size)
    _, _, fitness_curve_rhc_2 = mlrose_hiive.random_hill_climb(problem, max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)
    _, _, fitness_curve_sa_2 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)    
    _, _, fitness_curve_ga_2 = mlrose_hiive.genetic_alg(problem, max_attempts = 1000, max_iters = 10000, curve = True)
    _, _, fitness_curve_mimic_2 = mlrose_hiive.mimic(problem, pop_size = 500, max_attempts = 100, max_iters = 10000, curve = True)

    fitness = mlrose_hiive.FourPeaks(t_pct = 0.5)
    problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size = space_size)
    _, _, fitness_curve_rhc_3 = mlrose_hiive.random_hill_climb(problem, max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)
    _, _, fitness_curve_sa_3 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 1000, max_iters = 10000, init_state = init_state, curve = True)    
    _, _, fitness_curve_ga_3 = mlrose_hiive.genetic_alg(problem, max_attempts = 1000, max_iters = 10000, curve = True)
    _, _, fitness_curve_mimic_3 = mlrose_hiive.mimic(problem, pop_size = 500, max_attempts = 100, max_iters = 10000, curve = True)

    plt.figure()
    plt.subplot(311)
    plt.plot(fitness_curve_rhc_1[:,0], label = 'RHC, t = 0.1')
    plt.plot(fitness_curve_sa_1[:,0], label = 'SA, t = 0.1')    
    plt.plot(fitness_curve_ga_1[:,0], label = 'GA, t = 0.1')
    plt.plot(fitness_curve_mimic_1[:,0], label = 'MIMIC, t = 0.1')
    plt.title('Fitness Curve for Thresholds_4P')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.subplot(312)
    plt.plot(fitness_curve_rhc_2[:,0], label = 'RHC, t = 0.2')
    plt.plot(fitness_curve_sa_2[:,0], label = 'SA, t = 0.2')    
    plt.plot(fitness_curve_ga_2[:,0], label = 'GA, t = 0.2')
    plt.plot(fitness_curve_mimic_2[:,0], label = 'MIMIC, t = 0.2')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.subplot(313)
    plt.plot(fitness_curve_rhc_3[:,0], label = 'RHC, t = 0.5')
    plt.plot(fitness_curve_sa_3[:,0], label = 'SA, t = 0.5')    
    plt.plot(fitness_curve_ga_3[:,0], label = 'GA, t = 0.5')
    plt.plot(fitness_curve_mimic_3[:,0], label = 'MIMIC, t = 0.5')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig('4P_thresholds.png')

    #test different max attempts
    space_size = 50
    max_atp = 100
    fitness = mlrose_hiive.FourPeaks(t_pct = 0.1)
    problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size = space_size)
    _, _, fitness_curve_rhc_1 = mlrose_hiive.random_hill_climb(problem, max_attempts = max_atp, max_iters = 10000, init_state = init_state, curve = True)
    _, _, fitness_curve_sa_1 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = max_atp, max_iters = 10000, init_state = init_state, curve = True)    
    _, _, fitness_curve_ga_1 = mlrose_hiive.genetic_alg(problem, max_attempts = max_atp, max_iters = 10000, curve = True)
    _, _, fitness_curve_mimic_1 = mlrose_hiive.mimic(problem, pop_size = 500, max_attempts = max_atp/10, max_iters = 10000, curve = True)

    max_atp = 500
    fitness = mlrose_hiive.FourPeaks(t_pct = 0.1)
    problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size = space_size)
    _, _, fitness_curve_rhc_2 = mlrose_hiive.random_hill_climb(problem, max_attempts = max_atp, max_iters = 10000, init_state = init_state, curve = True)
    _, _, fitness_curve_sa_2 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = max_atp, max_iters = 10000, init_state = init_state, curve = True)    
    _, _, fitness_curve_ga_2 = mlrose_hiive.genetic_alg(problem, max_attempts = max_atp, max_iters = 10000, curve = True)
    _, _, fitness_curve_mimic_2 = mlrose_hiive.mimic(problem, pop_size = 500, max_attempts = max_atp/10, max_iters = 10000, curve = True)

    max_atp = 1000
    fitness = mlrose_hiive.FourPeaks(t_pct = 0.1)
    problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size = space_size)
    _, _, fitness_curve_rhc_3 = mlrose_hiive.random_hill_climb(problem, max_attempts = max_atp, max_iters = 10000, init_state = init_state, curve = True)
    _, _, fitness_curve_sa_3 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = max_atp, max_iters = 10000, init_state = init_state, curve = True)    
    _, _, fitness_curve_ga_3 = mlrose_hiive.genetic_alg(problem, max_attempts = max_atp, max_iters = 10000, curve = True)
    _, _, fitness_curve_mimic_3 = mlrose_hiive.mimic(problem, pop_size = 500, max_attempts = max_atp/10, max_iters = 10000, curve = True)

    plt.figure()
    plt.subplot(311)
    plt.plot(fitness_curve_rhc_1[:,0], label = 'RHC, atp = 100')
    plt.plot(fitness_curve_sa_1[:,0], label = 'SA, atp = 100')    
    plt.plot(fitness_curve_ga_1[:,0], label = 'GA, atp = 100')
    plt.plot(fitness_curve_mimic_1[:,0], label = 'MIMIC, atp = 10')
    plt.title('Fitness Curve for Attempts_4P')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.subplot(312)
    plt.plot(fitness_curve_rhc_2[:,0], label = 'RHC, atp = 500')
    plt.plot(fitness_curve_sa_2[:,0], label = 'SA, atp = 500')    
    plt.plot(fitness_curve_ga_2[:,0], label = 'GA, atp = 500')
    plt.plot(fitness_curve_mimic_2[:,0], label = 'MIMIC, atp = 50')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.subplot(313)
    plt.plot(fitness_curve_rhc_3[:,0], label = 'RHC, atp = 1000')
    plt.plot(fitness_curve_sa_3[:,0], label = 'SA, atp = 1000')    
    plt.plot(fitness_curve_ga_3[:,0], label = 'GA, atp = 1000')
    plt.plot(fitness_curve_mimic_3[:,0], label = 'MIMIC, atp = 100')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig('4P_attempts.png')

    
    #parametric runs for GA
    space_size = 50
    fitness = mlrose_hiive.FourPeaks(t_pct = 0.1)
    problem = mlrose_hiive.DiscreteOpt(length = space_size, fitness_fn = fitness, maximize = True, max_val = 2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size = space_size)
    _, _, fitness_curve_ga_1 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.1, pop_size = 100, max_attempts = 300, max_iters = 500, curve = True)
    _, _, fitness_curve_ga_2 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.25, pop_size = 100, max_attempts = 300, max_iters = 500, curve = True)
    _, _, fitness_curve_ga_3 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.1, pop_size = 200, max_attempts = 300, max_iters = 500, curve = True)
    _, _, fitness_curve_ga_4 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.25, pop_size = 200, max_attempts = 300, max_iters = 500, curve = True)
    _, _, fitness_curve_ga_5 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.1, pop_size = 500, max_attempts = 300, max_iters = 500, curve = True)
    _, _, fitness_curve_ga_6 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.25, pop_size = 500, max_attempts = 300, max_iters = 500, curve = True)

    plt.figure()
    plt.plot(fitness_curve_ga_1[:,0], label = 'mutation prob = 0.1, population = 100')
    plt.plot(fitness_curve_ga_2[:,0], label = 'mutation prob = 0.25, population = 100')
    plt.plot(fitness_curve_ga_3[:,0], label = 'mutation prob = 0.1, population = 200')
    plt.plot(fitness_curve_ga_4[:,0], label = 'mutation prob = 0.25, population = 200')
    plt.plot(fitness_curve_ga_5[:,0], label = 'mutation prob = 0.1, population = 500')
    plt.plot(fitness_curve_ga_6[:,0], label = 'mutation prob = 0.25, population = 500')
    plt.title('GA Analysis (Four Peaks)')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig('4P_ga param.png')
    

if __name__ == "__main__":
	main()
	
