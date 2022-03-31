import copy

from GAparsimony import Population, order
from GAparsimony.util import parsimony_monitor, parsimony_summary
from GAparsimony.lhs import geneticLHS, improvedLHS, maximinLHS, optimumLHS, randomLHS
import math
import numpy as np
import pandas as pd
import time
from numpy.random import multinomial

def parsimony_monitor(iter, fitnessval, bestfitnessVal, bestfitnessTst, bestcomplexity, minutes_gen, digits=7, *args):
    r"""Functions for monitoring GA-PARSIMONY algorithm evolution

    Functions to print summary statistics of fitness values at each iteration of a GA search.

    Parameters
    ----------
    object : object of GAparsimony
        The `GAparsimony` object that we want to monitor .
    digits : int
        Minimal number of significant digits.
    *args :
        Further arguments passed to or from other methods.
    """

    fitnessval = fitnessval[~np.isnan(fitnessval)]

    print(f"PSO-PARSIMONY | iter = {iter}")
    print("|".join([f"MeanVal = {round(np.mean(fitnessval), digits)}".center(16 + digits),
                    f"ValBest = {round(bestfitnessVal, digits)}".center(16 + digits),
                    f"TstBest = {round(bestfitnessTst, digits)}".center(16 + digits),
                    f"ComplexBest = {round(bestcomplexity, digits)}".center(19 + digits),
                    f"Time(min) = {round(minutes_gen, digits)}".center(17 + digits)]) + "\n")


def _population(pop, seed_ini, popSize, type_ini_pop="randomLHS", ):
    r"""
    Population initialization in GA-PARSIMONY with a combined chromosome of model parameters 
    and selected features. Functions for creating an initial population to be used in the GA-PARSIMONY process.

    Generates a random population of `GAparsimony.popSize` individuals. For each individual a 
    random chromosome is generated with `len(GAparsimony.population._params)` real values in the `range[GAparsimony._min, GAparsimony._max] `
    plus `len(GAparsimony.population.colsnames)` random binary values for feature selection. `random` or Latin Hypercube Sampling can 
    be used to create a efficient spread initial population.

    Parameters
    ----------
    type_ini_pop : list, {'randomLHS', 'geneticLHS', 'improvedLHS', 'maximinLHS', 'optimumLHS'}, optional
        How to create the initial population. `random` optiom initialize a random population between the 
        predefined ranges. Values `randomLHS`, `geneticLHS`, `improvedLHS`, `maximinLHS` & `optimumLHS` 
        corresponds with several meth-ods of the Latin Hypercube Sampling (see `lhs` package for more details).

    Returns
    -------
    numpy.array
        A matrix of dimension `GAparsimony.popSize` rows and `len(GAparsimony.population._params)+len(GAparsimony.population.colsnames)` columns.

    """

    nvars = len(pop._params) + len(pop.colsnames)
    if type_ini_pop == "randomLHS":
        population = randomLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "geneticLHS":
        population = geneticLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "improvedLHS":
        population = improvedLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "maximinLHS":
        population = maximinLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "optimumLHS":
        population = optimumLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "random":
        population = (np.random.rand(popSize * nvars) * (nvars - popSize) + popSize).reshape(
            popSize * nvars, 1)

    # Scale matrix with the parameters range
    population = population * (pop._max - pop._min)
    population = population + pop._min

    # If one wants to convert features to binary:
    #population[:, len(pop._params):nvars] = population[:,len(pop._params):nvars] <= feat_thres

    return population


def parsimony_summary(fitnessval, fitnesstst, complexity, *args):
    x1 = fitnessval[~np.isnan(fitnessval)]
    q1 = np.percentile(x1, [0, 25, 50, 75, 100])
    x2 = fitnesstst[~np.isnan(fitnesstst)]
    q2 = np.percentile(x1, [0, 25, 50, 75, 100])
    x3 = complexity[~np.isnan(complexity)]
    q3 = np.percentile(x1, [0, 25, 50, 75, 100])

    return q1[4], np.mean(x1), q1[3], q1[2], q1[1], q1[0], q2[4], np.mean(x2), q2[3], q2[2], q2[1], q2[0], q3[
        4], np.mean(x3), q3[3], q3[2], q3[1], q3[0]


def _rerank(fitnessval, complexity, popSize, rerank_error, preserve_best=True):
    r"""
    Function for reranking by complexity in parsimonious model selectionprocess. Promotes models with similar fitness but lower complexity to top positions.

    This method corresponds with the second step of parsimonious model selection (PMS) procedure.PMS works in the
    following way: in each GA generation, best solutions are first sorted by their cost,J. Then, in a second step,
    individuals with less complexity are moved to the top positions when theabsolute difference of their J is lower
    than aobject@rerank_errorthreshold value. Therefore, theselection of less complex solutions among those with similar
    accuracy promotes the evolution ofrobust solutions with better generalization capabilities.

    Returns
    -------
    numpy.array
        A vector with the new position of the individuals

    """

    cost1 = fitnessval.copy().astype(float)
    cost1[np.isnan(cost1)] = np.NINF

    sort = order(cost1, decreasing=True)
    cost1 = cost1[sort]
    complexity = complexity.copy()
    complexity[np.isnan(complexity)] = np.Inf
    complexity = complexity[sort]
    position = sort

    # start
    if preserve_best and len(cost1)>1:
        pos1 = 1
        pos2 = 2
        error_posic = cost1[1]
    else:
        pos1 = 0
        pos2 = 1
        error_posic = cost1[0]
    cambio = False

    while not pos1 == popSize:
        # Obtaining errors
        if pos2 >= popSize:
            if cambio:
                pos2 = pos1 + 1
                cambio = False
            else:
                break
        error_indiv2 = cost1[pos2]

        # Compare error of first individual with error_posic. Is greater than threshold go to next point
        #      if ((Error.Indiv1-error_posic) > model@rerank_error) error_posic=Error.Indiv1

        if np.isfinite(error_indiv2) and np.isfinite(error_posic):
            error_dif = abs(error_indiv2 - error_posic)
        else:
            error_dif = np.Inf
        if error_dif < rerank_error:

            # If there is not difference between errors swap if Size2nd < SizeFirst
            size_indiv1 = complexity[pos1]
            size_indiv2 = complexity[pos2]
            if size_indiv2 < size_indiv1:
                cambio = True

                swap_indiv = cost1[pos1]
                cost1[pos1] = cost1[pos2]
                cost1[pos2] = swap_indiv

                complexity[pos1], complexity[pos2] = complexity[pos2], complexity[pos1]

                position[pos1], position[pos2] = position[pos2], position[pos1]

                # if self.verbose == 2:
                #     print(f"SWAP!!: pos1={pos1}({size_indiv1}), pos2={pos2}({size_indiv2}), error_dif={error_dif}")
                #     print("-----------------------------------------------------")
            pos2 = pos2 + 1

        elif cambio:
            cambio = False
            pos2 = pos1 + 1
        else:
            pos1 = pos1 + 1
            pos2 = pos1 + 1
            error_dif2 = abs(cost1[pos1] - error_posic)
            if not np.isfinite(error_dif2):
                error_dif2 = np.Inf
            if error_dif2 >= rerank_error:
                error_posic = cost1[pos1]
    return position


class PSOparsimony(object):

    def __init__(self,
                 fitness,
                 params,
                 features,
                 type_ini_pop="improvedLHS",
                 npart=50,
                 maxiter=40,
                 early_stop=None,
                 Lambda=1.0,
                 c1 = 1/2 + math.log(2),
                 c2 = 1/2 + math.log(2),
                 IW_max=0.9,
                 IW_min=0.4,
                 K=3,
                 pmutation=None,
                 tol = 1e-4,
                 rerank_error=0.005,
                 keep_history = False,
                 feat_thres = 0.90,
                 best_global_thres = 1,
                 particles_to_delete=None,
                 seed_ini = None,
                 verbose=1):

        self.type_ini_pop = type_ini_pop
        self.fitness = fitness
        self.params = params
        self.features = features
        self.npart = npart
        self.maxiter = maxiter
        self.early_stop = maxiter if not early_stop else early_stop
        self.Lambda = Lambda
        self.c1 = c1
        self.c2 = c2
        self.IW_max = IW_max
        self.IW_min = IW_min
        self.K = K
        self.tol = tol

        self.rerank_error = rerank_error
        self.verbose = verbose
        self.seed_ini = seed_ini
        if pmutation is None:
            self.pmutation = 1/len(features)
        else:
            self.pmutation = pmutation

        self.feat_thres = feat_thres

        self.minutes_total = 0
        self.history = list()
        self.keep_history = keep_history

        # Percentage of particles that will be influenced by the best global of their neighbourhoods
        # (otherwise, they will be influenced by the best of the iteration in each neighbourhood)
        self.best_global_thres = best_global_thres


        if particles_to_delete is not None and len(particles_to_delete) < maxiter:
            # If the length of the particles to delete is lower than the iterations, the array is completed with zeros
            # up to the number of iterations.
            self.particles_to_delete = np.zeros(maxiter).astype(int)
            self.particles_to_delete[:len(particles_to_delete)] = particles_to_delete[:]
        else:
            self.particles_to_delete = particles_to_delete

        if self.seed_ini:
            np.random.seed(self.seed_ini)



    def fit(self, X, y, iter_ini=0):

        if self.seed_ini:
            np.random.seed(self.seed_ini)

        population = Population(self.params, columns=self.features)
        population.population = _population(population, seed_ini=self.seed_ini, popSize=self.npart,
                                            type_ini_pop=self.type_ini_pop)  # To create the initial population


        # Update population to satisfy the feat_thres
        population.update_to_feat_thres(self.npart, self.feat_thres)


        nfs = len(population.colsnames)
        self._summary = np.empty((self.maxiter, 6 * 3,))
        self._summary[:] = np.nan
        self.best_score = np.NINF
        self.best_complexity = np.Inf

        maxFitness = np.Inf
        best_fit_particle = np.empty(self.npart)
        best_fit_particle[:] = np.NINF

        best_pos_particle = np.empty(shape=(self.npart, len(population._params) + nfs))
        best_complexity_particle = np.empty(self.npart)  # Complexities
        best_complexity_particle[:] = np.Inf

        range_numbers = population._max - population._min
        vmax = self.Lambda * range_numbers
        range_as_pd = pd.Series(range_numbers)
        lower_as_pd = pd.Series(population._min)
        v_norm = randomLHS(self.npart, len(population._params) + nfs)
        v_norm = pd.DataFrame(v_norm)
        v_norm = v_norm.apply(lambda row: row * range_as_pd, axis=1)
        v_norm = v_norm.apply(lambda row: row + lower_as_pd, axis=1)

        velocity = (v_norm - population._pop) / 2
        velocity = velocity.to_numpy()

        self.bestSolList = list()
        self.best_models_list = list()
        self.best_models_conf_list = list()


        # Variables to store the best global positions, fitnessval and complexity of each particle
        bestGlobalPopulation = copy.deepcopy(population._pop)
        bestGlobalFitnessVal = np.empty(self.npart)
        bestGlobalFitnessVal[:] = np.NINF
        bestGlobalComplexity = np.empty(self.npart)
        bestGlobalComplexity[:] = np.inf

        #Variable that tracks the deleted particles (their number in the table)
        deleted_particles = []
        valid_particles = [x for x in range(self.npart) if
                           x not in deleted_particles]  # valid particles (numbers in the table)

        fitnessval = np.empty(self.npart)
        fitnessval[:] = np.nan
        fitnesstst = np.empty(self.npart)
        fitnesstst[:] = np.nan
        complexity = np.empty(self.npart)
        complexity[:] = np.nan
        _models = np.empty(self.npart).astype(object)

        update_neighbourhoods = False

        for iter in range(self.maxiter):

            self.iter = iter

            tic = time.time()
            #####################################################
            # Compute solutions
            #####################################################


            for t in valid_particles:
                c = population.getChromosome(t)

                if np.sum(c.columns) > 0:
                    fit = self.fitness(c, X=X, y=y)
                    fitnessval[t] = fit[0][0]
                    fitnesstst[t] = fit[0][1]
                    complexity[t] = fit[0][2]
                    _models[t] = fit[1]


            if self.seed_ini:
                np.random.seed(self.seed_ini * iter)

            # Sort by the Fitness Value
            # ----------------------------
            sort = order(fitnessval, kind='heapsort', decreasing=True, na_last=True)

            PopSorted = population[sort, :]
            FitnessValSorted = fitnessval[sort]
            FitnessTstSorted = fitnesstst[sort]
            ComplexitySorted = complexity[sort]
            _modelsSorted = _models[sort]


            if self.verbose == 2:
                print("\nStep 1. Fitness sorted")
                print(np.c_[FitnessValSorted, FitnessTstSorted, ComplexitySorted, population.population][:10, :])
                # input("Press [enter] to continue")

            if self.rerank_error != 0.0:
                ord_rerank = _rerank(FitnessValSorted, ComplexitySorted, self.npart, self.rerank_error)
                PopSorted = PopSorted[ord_rerank]
                FitnessValSorted = FitnessValSorted[ord_rerank]
                FitnessTstSorted = FitnessTstSorted[ord_rerank]
                ComplexitySorted = ComplexitySorted[ord_rerank]
                _modelsSorted = _modelsSorted[ord_rerank]

                if self.verbose == 2:
                    print("\nStep 2. Fitness reranked")
                    print(np.c_[FitnessValSorted, FitnessTstSorted, ComplexitySorted, population.population][:10, :])
                    # input("Press [enter] to continue")


            # Keep results
            # ---------------
            self._summary[iter, :] = parsimony_summary(FitnessValSorted, FitnessTstSorted, ComplexitySorted)

            # Keep Best Solution of this iteration
            # ------------------
            bestfitnessVal = FitnessValSorted[0]
            bestfitnessTst = FitnessTstSorted[0]
            bestcomplexity = ComplexitySorted[0]
            bestIterSolution = np.concatenate([[bestfitnessVal, bestfitnessTst, bestcomplexity], PopSorted[0]])
            self.bestSolList.append(bestIterSolution)
            self.best_models_list.append(_modelsSorted[0])
            self.best_models_conf_list.append(PopSorted[0])

            # Keep Global Best Model
            # ------------------
            # The best_score of the whole process. It is update if we find a better score, or equal but with lower complexity.
            if bestfitnessVal > self.best_score or (bestfitnessVal == self.best_score and bestcomplexity < self.best_complexity):
                self.best_score = bestfitnessVal
                self.best_complexity = bestcomplexity
                self.bestsolution = bestIterSolution
                self.solution_best_score = np.r_[self.best_score,
                                            bestfitnessVal,
                                            bestfitnessTst,
                                            bestcomplexity]
                self.best_model = _modelsSorted[0]
                self.best_model_conf = PopSorted[0].copy()

            # Update global best positions, fitness and complexity of each particle (with NO rerank)
            for i in range(self.npart):
                if fitnessval[i] > bestGlobalFitnessVal[i] or (fitnessval[i] == bestGlobalFitnessVal[i] and complexity[i] < bestGlobalComplexity[i]):
                    bestGlobalPopulation[i,:] = population._pop[i,:]
                    bestGlobalFitnessVal[i] = fitnessval[i]
                    bestGlobalComplexity[i] = complexity[i]


            # Keep elapsed time in minutes
            # ----------------------------
            tac = time.time()
            elapsed_gen = (tac - tic) / 60.0
            self.minutes_total += + elapsed_gen

            # Keep this generation into the History list (with no order)
            # ------------------------------------------
            if self.keep_history:
                self.history.append(
                    pd.DataFrame(np.c_[population.population, fitnessval, fitnesstst, complexity],
                                 columns=list(population._params.keys()) + population.colsnames + ["fitnessval", "fitnesstst",
                                                                                                   "complexity"]))


            # Call to 'monitor' function
            # --------------------------
            if self.verbose > 0:
                parsimony_monitor(iter, fitnessval, bestfitnessVal, bestfitnessTst, bestcomplexity, elapsed_gen)

            if self.verbose == 2:
                print("\nStep 3. Fitness results")
                print(np.c_[FitnessValSorted, FitnessTstSorted, ComplexitySorted, population.population][:10, :])
                # input("Press [enter] to continue")

            # Exit?
            # -----
            best_val_cost = self._summary[:, 0][~np.isnan(self._summary[:, 0])]
            if bestfitnessVal >= maxFitness:
                break
            if iter == self.maxiter:
                break
            if (len(best_val_cost) - (np.min(np.arange(len(best_val_cost))[best_val_cost >= (np.max(best_val_cost) - self.tol)]))) >= self.early_stop:
                break

            ####################################################
            # Deletion step (disabled by default)
            ####################################################
            if self.particles_to_delete is not None and self.particles_to_delete[iter]>0:
                # particles_to_delete[iter] contains the number of particles to be deleted in that iteration
                # We delete the worse particles at that point (in global, not in that iteration).
                sort = order(bestGlobalFitnessVal, kind='heapsort', decreasing=True, na_last=True)
                sort_not_deleted = [x for x in sort if x not in deleted_particles]
                deleted_particles = deleted_particles + sort_not_deleted[-self.particles_to_delete[iter]:]
                valid_particles = [x for x in range(self.npart) if x not in deleted_particles]
                update_neighbourhoods = True

            #####################################################
            # Generation of the Neighbourhoods
            #####################################################
            # If there is no improvement, the neighbourhood is changed. It also changes if particles have been deleted.
            if FitnessValSorted[0] <= self.best_score or update_neighbourhoods:
                update_neighbourhoods = False
                nb = list()
                for i in range(self.npart):
                    # Each particle informs at random K particles (the same particle may be chosen several times), and informs itself.
                    # The parameter K is usually set to 3. It means that each particle informs at less one particle (itself), and at most K+1 particles (including itself)

                    # Thus, a random integer vector of K elements between 0 and npart-1 is created and we append the particle.
                    # Duplicates are removed and this represents the neighbourhood.
                    if i not in deleted_particles:
                        #nb.append(np.unique(np.append(np.random.randint(low=0, high=self.npart - 1, size=self.K), i)))

                        indices = np.random.randint(low=0, high=len(valid_particles), size=self.K) # High is not included
                        random_particles = [valid_particles[index] for index in indices]
                        nb.append(np.unique(np.append(random_particles, i)))
                    else:
                        nb.append(np.nan)

                # Create an array to decide if a particle must be influenced by the best global of the neighbourhoods or the best of the iteration
                nb_global = np.random.choice(a=[True, False], size=(self.npart,), p=[self.best_global_thres, 1-self.best_global_thres])



            ###########################################
            # Update particular global bests (best position of the particle in the whole process, wrt to rerank)
            ###########################################

            # We have to take care to avoid problems with rerank:
            # EXAMPLE (rerank = 0.08):
            # SCORE 0.80 0.85 0.90
            # COST    10  100  200
            # The best score wrt to rerank should be 0.85. But if we get 0.80 with cost 10 in the next
            # iteration, that would be chosen. This is wrong, since we would be moving to worse scores. The
            # rerank must be applied wrt the best global score of each particle.
            for t in range(self.npart):
                # Three cases:
                # (1) If the best improves much, then update.
                # (2) If the best does not improve much, but the complexity is lower, then update.
                # (3) Otherwise, rerank criterion, but "consuming the rerank" wrt to the global best.
                if (fitnessval[t] > best_fit_particle[t] + self.rerank_error) \
                    or (fitnessval[t] >= best_fit_particle[t] and complexity[t] < best_complexity_particle[t]) \
                    or (best_fit_particle[t] - fitnessval[t]) <= self.rerank_error - (bestGlobalFitnessVal[t] - best_fit_particle[t]) and complexity[t] < best_complexity_particle[t]:
                    best_fit_particle[t] = fitnessval[t]  # Update the particular best fit of that particle.
                    best_pos_particle[t, :] = population._pop[t, :]  # Update the particular best pos of that particle.
                    best_complexity_particle[t] = complexity[t] # Update the complexity (could be more complex if the fitnessval[t] is better)

            ###########################################
            # Compute Local bests in the Neighbourhoods
            ###########################################
            best_pos_neighbourhood = np.empty(shape=(self.npart, len(population._params) + nfs))  # Matrix in which i-th row contains the best particle of the i-th neighbourhood.
            best_fit_neighbourhood = np.empty(self.npart)  # Array that contains in position i the score of the best particle of the i-th neighbourhood.
            best_fit_neighbourhood[:] = np.Inf

            for i in valid_particles:

                if nb_global[i]: # If the global best of the neighbourhood must be selected
                    particles_positions = nb[i]  # Positions of the neighbourhood particles (number within population)
                    local_fits = best_fit_particle[particles_positions]
                    local_complexity = best_complexity_particle[particles_positions]
                    local_sort = order(local_fits, kind='heapsort', decreasing=True, na_last=True)
                    local_fits_sorted = local_fits[local_sort]
                    local_complexity_sorted = local_complexity[local_sort]
                    local_sort_rerank = _rerank(local_fits_sorted, local_complexity_sorted, len(local_fits),
                                                self.rerank_error, preserve_best=True)
                    max_local_fit_pos = particles_positions[local_sort[local_sort_rerank[0]]]
                    best_pos_neighbourhood[i, :] = best_pos_particle[max_local_fit_pos, :]
                    #best_fit_neighbourhood[i] = best_fit_particle[max_local_fit_pos]

                else: # The best of the neighbourhood in the current iteration
                    particles_positions = nb[i]  # Positions of the neighbourhood particles (number within population)
                    local_fits = fitnessval[particles_positions]

                    local_complexity = complexity[particles_positions]
                    local_sort = order(local_fits, kind='heapsort', decreasing=True, na_last=True)
                    local_fits_sorted = local_fits[local_sort]
                    local_complexity_sorted = local_complexity[local_sort]
                    local_sort_rerank = _rerank(local_fits_sorted,local_complexity_sorted, len(local_fits), self.rerank_error, preserve_best=False)
                    max_local_fit_pos = particles_positions[local_sort[local_sort_rerank[0]]]

                    best_pos_neighbourhood[i, :] = population._pop[max_local_fit_pos, :]
                    #best_fit_neighbourhood[i] = fitnessval[max_local_fit_pos]


            #####################################################
            # Update positions and velocities following SPSO 2007
            #####################################################

            U1 = np.random.uniform(low=0, high=1,
                                   size=(self.npart, len(population._params) + nfs))  # En el artículo se llaman r1 y r2
            U2 = np.random.uniform(low=0, high=1,
                                   size=(self.npart, len(population._params) + nfs))  # En el artículo se llaman r1 y r2

            IW = self.IW_max - (self.IW_max - self.IW_min) * iter / self.maxiter

            # Two first terms of the velocity
            velocity = IW * velocity + U1 * self.c1 * (best_pos_particle - population._pop)

            velocity[:, :] = velocity[:, :] + self.c2 * U2[:, :] * (
                        best_pos_neighbourhood[:, :] - population._pop[:, :])

            # Limit velocity to vmax to avoid explosion

            for j in range(len(population._params) + nfs):
                vmax_pos = np.where(abs(velocity[:,j]) > vmax[j])[0]
                for i in vmax_pos:
                    velocity[i, j] = math.copysign(1, velocity[i, j]) * abs(vmax[j])

            ##############################
            # Update positions of FEATURES
            ##############################
            nparams = len(population._params)
            for nf in range(nparams,nparams + nfs): # We must move to the features (the particles contain first hyper-parameters and then features)
                for p in range(self.npart):
                    population._pop[p,nf] = population._pop[p,nf] + velocity[p,nf] # Update positions for the model positions (x = x + v)
                    # To ensure that the interval [0,1] is preserved
                    if population._pop[p, nf] > 1.0:
                        population._pop[p, nf] = 1.0
                    if population._pop[p,nf] < 0.0:
                        population._pop[p, nf] = 0.0

            ######################
            # Mutation of FEATURES
            # ####################
            if self.pmutation > 0:
                rnd_mut = np.random.uniform(size = (self.npart, nfs))
                for p in range(self.npart):
                    for nf in range(nparams,nparams + nfs):
                        if rnd_mut[p, nf - nparams] < self.pmutation:
                            if population._pop[p, nf] < 0.5:
                                population._pop[p, nf] = np.random.uniform(low=0.5, high=1.0)
                            else:
                                population._pop[p, nf] = np.random.uniform(low=0.0, high=0.5)


            #######################################################
            # Update positions of model HYPERPARAMETERS (x = x + v)
            #######################################################

            for j in range(nparams):
                population._pop[:, j] = population._pop[:, j] + velocity[:, j]

            ################################################################################################
            # Confinement Method for SPSO 2007 - absorbing2007 (hydroPSO) - Deterministic Back (Clerc, 2007)
            ################################################################################################
            for j in range(nparams):
                out_max = (population._pop[:, j] > population._max[j])
                out_min = (population._pop[:, j] < population._min[j])
                population._pop[out_max, j] = population._max[j]
                population._pop[out_min, j] = population._min[j]
                velocity[out_max, j] = 0
                velocity[out_min, j] = 0
