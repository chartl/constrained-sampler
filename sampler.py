# -*- coding: utf-8 -*- 
"""\
Approximate homogenous sampling of a constrained volume

"""
from __future__ import print_function
from __future__ import division

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import scipy as sp
import theano
import theano.tensor as tt
import pymc3 as pm



TARGET_OVERSAMPLING = 20. # for initial landmarks, want to oversample by a factor of 20


def get_args():
    parser = ArgumentParser('sampler')
    parser.add_argument('input_file', help='The input file listing dimensionality and constraints')
    parser.add_argument('output_file', help='The output file to which to write sampled vectors')
    parser.add_argument('n_results', help='The number of points to sample in the space', type=int)
    parser.add_argument('--no_filter_warnings', help='Do not turn off theano warnings', action='store_true')

    return parser.parse_args()


def str2fx(xstr):
    """\
    Given a stringified constraint such as
    
    1 - x[0] - x[1]
    
    convert it to a callable function
    
    """
    def f(x):
        return eval(xstr, {'__builtins__': None}, {'x': x } )
    return f


def read_constraints(fn):
    """\
    Grab the dimensionality and constraints from a given file. 
    
    """
    gs = list()
    dim = 0
    for i, line in enumerate(open(fn)):
        if dim is 0:
            dim = int(line.strip())
        elif i == 1:
            assert len(line.strip().split()) == dim, 'Example point does not match dimensionality'
        elif line[0] != '#' and '>=' in line:
            gs.append(line.strip().split('>=')[0].strip())
    return {'dim': dim, 
            'constraints': gs, 
            'constraints_fx': [str2fx(c) for c in gs]}


def subset_accepting(X, constraint_fx):
    """\
    Given a set of points, and constrants g(x) [as callable functions], subset just to the
    points that fall into the valid volume
    
    """
    constraint_mat = np.vstack([np.apply_along_axis(c, 1, X) < 0 for c in constraint_fx])
    in_region = np.logical_not(np.apply_along_axis(np.any, 0, constraint_mat))
    return X[in_region, :]


def uniform(dim, const_str, const_fx, K, *args):
    """\
    Baseline uniform sampling on the hypercube

    """
    return np.random.uniform(size=(K, dim))


def MCMC(dim, const_str, const_fx, K, chains=3, cores=3):
    """\
    MCMC sampling, with potentials lowering the probability
    of drawing failing points

    """
    lambda_values = [1, 10, 20]
    k = int(K/(chains*len(lambda_values)))
    Xvals = list()
    for lam in lambda_values:
        with pm.Model() as mod:
            x = pm.Uniform('x', shape=dim)
            for i, const in enumerate(const_str):
                cname = 'g%d' % i
                g = pm.Deterministic(cname, eval(const))
                pname = '%s_pot' % cname
                pm.Potential(pname, tt.switch(tt.lt(g, 0), lam*g, 0))
            trace = pm.sample(k, tune=1000, chains=chains, cores=cores)
        Xvals.append(trace['x'])
    return np.vstack(Xvals)


def VINormal(dim, const_str, const_fx, K, nfit=30000):
    """\
    Normal (full-rank) sampling, fit with ADVI to a
    high-potential probability distribution

    """
    with pm.Model() as mod:
        x = pm.Uniform('x', shape=dim)
        for i, const in enumerate(const_str):
            cname = 'g%d' % i
            g = pm.Deterministic(cname, eval(const))
            pname = '%s_pot' % cname
            pm.Potential(pname, tt.switch(tt.lt(g, 0), 7500*g, 0))
        fit_res = pm.fit(nfit, method='fullrank_advi', obj_n_mc=3)
        trace = fit_res.sample(K)
    return trace['x']


def switch_method(params, k_wanted):
    Xtest = uniform(params['dim'], params['constraints'], params['constraints_fx'], K=10000)
    Xtest = subset_accepting(Xtest, params['constraints_fx'])
    p_accept = Xtest.shape[0]/10000.
    if p_accept == 0.:
        print('Warning: Acceptance volume < 0.1%; region may be infeasible\n   Method: VI')
        return VINormal
    else:
        print('Estimated volume: %.2f%%' % (100*p_accept))
        K_to_sample = k_wanted * TARGET_OVERSAMPLING / p_accept
        if K_to_sample < 10**5:
            print('  Method: Uniform')
            return uniform
        elif K_to_sample < 10**7:
            print('  Method: MCMC')
            return MCMC
        print('  Method: VI')
        return VINormal


def sample_until(sampler, params, K):
    X = sampler(params['dim'], params['constraints'], params['constraints_fx'], K*2)
    X_acc = subset_accepting(X, params['constraints_fx'])
    while X_acc.shape[0] < 0.8*K:
        X_new = sampler(params['dim'], params['constraints'], params['constraints_fx'], K)
        X_acc = np.vstack([X_acc, subset_accepting(X_new, params['constraints_fx'])])
    return X_acc


def get_landmarks(X, constraint_fx, forget=3, k_landmarks=1000):
    Xp = subset_accepting(X, constraint_fx)
    landmark_set = list()
    landmark_dist = np.zeros((Xp.shape[0],))
    # initialize
    idx = np.random.randint(Xp.shape[0])
    landmark_dist = np.linalg.norm(Xp-Xp[idx,:], axis=1)
    for _ in range(forget):
        idx = np.argmax(landmark_dist)
        landmark_dist = np.linalg.norm(Xp-Xp[idx,:], axis=1)
    landmark_set.append(idx)
    for j in range(1, k_landmarks):
        idx = np.argmax(landmark_dist)
        landmark_dist = np.minimum(landmark_dist, np.linalg.norm(Xp-Xp[idx,:],axis=1))
        landmark_set.append(idx)
    return Xp[np.array(landmark_set),:]



def meddist(X_pass):
    return np.median(sp.spatial.distance_matrix(X_pass, X_pass))



def as_theano_f(const):
    const = const.replace('[', '[:,')
    def g(Xo):
        return eval(const, {'__builtins__': None}, {'x': Xo} )
    return g


def refine_landmarks(X, constraint_str, iters):
    theano.config.compute_test_value = 'ignore'
    Xi = tt.matrix('base_input')
    Delta = theano.shared(0*X)
    Xo = Xi + Delta
    dif = Xo.reshape((Xo.shape[0], 1, -1)) - Xo.reshape((1, Xo.shape[0], -1))
    cost_dist = -(dif**2).sum()
    cost = cost_dist
    for const in constraint_str:
        g = as_theano_f(const)(Xo)
        cost_g = 1/(4*g)
        cost += cost_g.sum()
    # don't forget to add in the hypercube constraints!
    for j in range(X.shape[1]):
        h = Xo[:,j]
        h2 = 1 - Xo[:,j]
        cost_h = 1/(4*h) + 1/(4*h2)
        cost += cost_h.sum()
    dim_adj = min(1., 4./X.shape[1])
    # ideally this would be some more automatic method like ADAM
    alpha = np.var(X)*1e-7*dim_adj**2 # smaller learning rate in higher dimensions and narrower regions
    grad_updates = [(Delta, Delta - alpha * theano.grad(cost, Delta))]
    train = theano.function([Xi],cost_dist,updates=grad_updates)
    for _ in range(iters):
        energy = train(X)
        if ( _ % (int(iters/5)) == 0 ):
            print('Energy: %.2e' % energy)
    to_ret = theano.function([Xi], Xo)(X)
    theano.config.compute_test_value = 'raise'
    return to_ret


def main(args):
    if not args.no_filter_warnings:
        import warnings
        warnings.filterwarnings("ignore")
    space_info = read_constraints(args.input_file)
    method = switch_method(space_info, args.n_results)
    init_points = sample_until(method, space_info, 1 + int(args.n_results * TARGET_OVERSAMPLING))
    success, landmark_rate, l_iter = False, 1.01, 100 * min(1, 4./init_points.shape[1])
    raw_landmarks = get_landmarks(init_points, space_info['constraints_fx'], k_landmarks=args.n_results)
    while not success:
        landmarks = get_landmarks(init_points, space_info['constraints_fx'], k_landmarks=1+int(landmark_rate*args.n_results))
        landmarks = refine_landmarks(landmarks, space_info['constraints'], 1 + int(l_iter))
        landmarks = get_landmarks(landmarks, space_info['constraints_fx'], k_landmarks=args.n_results)
        landmark_rate = landmark_rate * 1.01
        success = landmarks.shape[0] == args.n_results
    H_raw, H_lm = meddist(raw_landmarks), meddist(landmarks)
    print('Raw samples: %.3e, Refined: %.3e (delta: %.2f%%)' % (H_raw, H_lm, 100*(H_lm-H_raw)/H_lm))
    data = pd.DataFrame(landmarks) if H_lm > H_raw else pd.DataFrame(raw_landmarks)
    data.columns = ['X%d' % j for j in range(data.shape[1])]
    data.to_csv(args.output_file, sep='\t', index=False)


if __name__ == '__main__':
    main(get_args())


