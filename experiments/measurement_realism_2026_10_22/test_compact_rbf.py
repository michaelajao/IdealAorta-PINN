import importlib.util
import unittest
import numpy as np
from compact_rbf import pair_kernel,CompactRBF,fit_compact_pair,separated_centres,solve_regularized
from observations import Observations
from operators import voxel_quadrature


class CompactKernelTests(unittest.TestCase):
    def test_kernel_matches_independent_scalar_hessian(self):
        def phi(x):
            r=np.linalg.norm(x);return max(1-r,0)**6*(35*r*r+18*r+3)/3
        for x in [np.array([.2,.3,.1]),np.array([-.1,.2,-.4])]:
            h=5e-5;H=np.empty((3,3))
            for i in range(3):
                ei=np.eye(3)[i]*h
                H[i,i]=(phi(x+ei)-2*phi(x)+phi(x-ei))/h**2
                for j in range(i):
                    ej=np.eye(3)[j]*h
                    H[i,j]=H[j,i]=(phi(x+ei+ej)-phi(x+ei-ej)-phi(x-ei+ej)+phi(x-ei-ej))/(4*h*h)
            expected=3/112*(H-np.trace(H)*np.eye(3))
            np.testing.assert_allclose(pair_kernel(x[None],1,True)[0],expected,atol=2e-7)

    def test_origin_cutoff_and_global_divergence(self):
        np.testing.assert_array_equal(pair_kernel(np.zeros((1,3)),1,True)[0],np.eye(3))
        np.testing.assert_array_equal(pair_kernel([[1,0,0],[1.1,0,0]],1,True),np.zeros((2,3,3)))
        self.assertEqual(pair_kernel([[0,0,0]],1,False)[0],1)
        rng=np.random.default_rng(18);centres=rng.uniform(-.5,.5,(5,3));coef=rng.normal(size=(5,3))
        x=rng.uniform(-.8,.8,(20,3))
        def f(p):
            k=pair_kernel((p[:,None]-centres).reshape(-1,3),.8,True).reshape(len(p),5,3,3)
            return np.einsum('nmij,mj->ni',k,coef)
        errors=[]
        for eps in [1e-4,5e-5]:
            divergence=np.zeros(len(x))
            for c in range(3):
                d=np.eye(3)[c]*eps
                divergence+=(f(x+d)[:,c]-f(x-d)[:,c])/(2*eps)
            errors.append(np.linalg.norm(divergence))
        self.assertLess(errors[1],errors[0]*.27)
        self.assertLess(errors[1],1e-4)


@unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'sparse compact fits verified on Brosnan')
class SparseCompactTests(unittest.TestCase):
    def test_direct_measurements_only_fits_without_implicit_boundary_targets(self):
        rng=np.random.default_rng(396);points=rng.uniform(-1,1,(45,3))
        def field(x):return np.column_stack([x[:,0]+2*x[:,1]+1,-x[:,1]+x[:,2],x[:,0]])
        support,weights=voxel_quadrature(points,.05,2)
        values=np.stack([field(points),2*field(points)])
        validation=np.zeros(len(points),bool);validation[-6:]=True
        obs=Observations.make([0,10],support,weights,values,validation,np.arange(len(points)),0,1)
        empty=np.empty((0,3));inlet=np.empty((2,0,3))
        for vector in (False,True):
            models,receipt=fit_compact_pair(obs,empty,empty,inlet,vector,[2.],[1e-9],
                measurements_only=True,solver_method='direct',direct_block_columns=13)
            # A kernel+affine ridge fit need not recover the exact affine field
            # at held-out locations. Compare ALL predictions with an independent
            # augmented least-squares reference for the declared fitting data.
            design=models[0].design(support[~validation],weights).toarray()/np.sqrt((~validation).sum())
            query=models[0].design(support,weights).toarray();n=design.shape[1]
            augmented=np.vstack([design,np.sqrt(1e-9)*np.eye(n)])
            for frame,model in enumerate(models):
                target=values[frame,~validation]
                if vector:target=target.reshape(-1)
                zeros=np.zeros(n) if vector else np.zeros((n,3))
                expected=np.linalg.lstsq(augmented,np.concatenate([target/np.sqrt((~validation).sum()),zeros]),rcond=None)[0]
                expected_prediction=(query@expected).reshape(-1,3)
                np.testing.assert_allclose(model.observe(support,weights),expected_prediction,atol=2e-6,rtol=2e-6)
            np.testing.assert_allclose(models[0].observe(support[~validation],weights),values[0,~validation],atol=2e-6)
            self.assertEqual(receipt['n_fitting_wall_nodes'],0)
            self.assertEqual(receipt['n_fitting_inlet_nodes'],0)
            self.assertTrue(receipt['boundary_contract'].startswith('measurements only'))
            self.assertTrue(receipt['trials'][0]['direct_solver']['converged'])

    def test_direct_pair_matches_iterative_fit_for_both_families_and_observation_operators(self):
        rng=np.random.default_rng(394);x=rng.uniform(-.3,.3,(35,3));centres=x[:4]
        truth=CompactRBF(centres,.45,True,rng.normal(size=3*len(centres)+11))
        wall=rng.uniform(1,2,(8,3));inlet=rng.uniform(-2,-1,(4,3))
        validation=np.zeros(len(x),bool);validation[-6:]=True
        for averaged in (False,True):
            support,weights=voxel_quadrature(x,.04,2) if averaged else (x[:,None],np.ones(1))
            values=truth.observe(support,weights)
            obs=Observations.make([0,10],support,weights,np.stack([values,2*values]),validation,
                                  np.arange(len(x)),.001,15)
            before=obs.values.copy();inlet_values=np.stack([truth.predict(inlet),2*truth.predict(inlet)])
            for vector in (False,True):
                reference,_=fit_compact_pair(obs,wall,inlet,inlet_values,vector,[.45],[.001],
                                             tol=1e-11,max_iter=15000)
                models,record=fit_compact_pair(obs,wall,inlet,inlet_values,vector,[.45],[.001],
                    tol=1e-11,solver_method='direct',direct_block_columns=7)
                for a,b in zip(models,reference):
                    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=3e-7,rtol=3e-7)
                self.assertEqual(record['solver_method'],'direct')
                self.assertIsNone(record['trials'][0]['lsmr'])
                self.assertTrue(record['trials'][0]['direct_solver']['converged'])
                self.assertFalse(record['selection_uses_hidden_targets'])
                np.testing.assert_array_equal(obs.values,before)

    def test_measurements_only_comparators_fit_observed_affine_without_fake_boundaries(self):
        rng=np.random.default_rng(240);points=rng.uniform(-1,1,(50,3))
        def field(x):return np.column_stack([x[:,0]+2*x[:,1]+1,-x[:,1]+x[:,2],x[:,0]])
        support,weights=voxel_quadrature(points,.05,2)
        values=np.stack([field(points),2*field(points)])
        validation=np.zeros(50,bool);validation[-5:]=True
        obs=Observations.make([0,10],support,weights,values,validation,np.arange(50),0,1)
        empty=np.empty((0,3));inlet_values=np.empty((2,0,3))
        for vector in (False,True):
            models,receipt=fit_compact_pair(obs,empty,empty,inlet_values,vector,[2.],[0],
                tol=1e-11,max_iter=3000,measurements_only=True)
            np.testing.assert_allclose(models[0].observe(support[~validation],weights),values[0,~validation],atol=1e-6)
            self.assertTrue(receipt['boundary_contract'].startswith('measurements only'))
        with self.assertRaisesRegex(ValueError,'native fitting requires'):
            fit_compact_pair(obs,empty,empty,inlet_values,True,[2.],[0])
        with self.assertRaisesRegex(ValueError,'boundary targets or a boundary gate'):
            fit_compact_pair(obs,empty,empty,inlet_values,True,[2.],[0],measurements_only=True,boundary_tolerance=.02)
    def test_column_scaling_preserves_original_ridge_objective(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(232)
        a=rng.normal(size=(50,12))*np.geomspace(.01,10,12)
        b=rng.normal(size=50);ridge=.003
        expected=np.linalg.solve(a.T@a+ridge*np.eye(12),a.T@b)
        for scaled in (False,True):
            actual,record=solve_regularized(csr_matrix(a),b,ridge,1e-12,2000,scaled)
            np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8)
            self.assertIn(record['istop'],(1,2,4,5))
            if scaled:self.assertLess(record['original_stationarity_Frobenius_normalized'],1e-10)
    def test_basis_thinning_preserves_coverage_and_positive_separation(self):
        from scipy.spatial import cKDTree
        rng=np.random.default_rng(229);points=rng.uniform(-1,1,(300,3));points=np.vstack([points,points[:20]])
        before=points.copy();selected=separated_centres(points,.2)
        self.assertLess(len(selected),len(points))
        self.assertLessEqual(cKDTree(selected).query(points)[0].max(),.2+1e-14)
        self.assertGreater(cKDTree(selected).query(selected,k=2)[0][:,1].min(),.2-1e-14)
        np.testing.assert_array_equal(points,before)
        np.testing.assert_array_equal(separated_centres(points,0),points)

    def test_known_inlet_failure_cannot_win_observed_holdout_selection(self):
        rng=np.random.default_rng(231);x=rng.uniform(-.2,.2,(20,3))
        validation=np.zeros(20,bool);validation[-4:]=True
        values=np.zeros((2,20,3));values[:,:,0]=1.
        obs=Observations.make([0,10],x[:,None],np.ones(1),values,validation,np.arange(20),0,1)
        wall=np.array([[1.,0,0],[1,1,0]])
        inlet=np.array([[-1.,0,0],[-1,1,0]])
        inlet_values=np.zeros((2,2,3));inlet_values[:,:,0]=1.
        with self.assertRaisesRegex(RuntimeError,'boundary-admissible'):
            fit_compact_pair(obs,wall,inlet,inlet_values,True,[.3],[1e8],max_iter=500,
                             boundary_tolerance=.02)

    def test_sparse_average_design_matches_direct_sum_and_guard(self):
        rng=np.random.default_rng(4);x=rng.uniform(-.5,.5,(8,3));centres=rng.uniform(-.5,.5,(6,3))
        p,w=voxel_quadrature(x,.2,2)
        for vector in [False,True]:
            model=CompactRBF(centres,.9,vector,polynomial=False)
            if vector:
                dense=sum(weight*pair_kernel((p[:,q,None]-centres).reshape(-1,3),.9,True).reshape(8,6,3,3)
                          for q,weight in enumerate(w))
                expected=dense.transpose(0,2,1,3).reshape(24,18)
            else:
                expected=sum(weight*pair_kernel((p[:,q,None]-centres).reshape(-1,3),.9,False).reshape(8,6)
                             for q,weight in enumerate(w))
            np.testing.assert_allclose(model.design(p,w).toarray(),expected,atol=1e-14)
            with self.assertRaises(MemoryError):model.design(p,w,max_nnz=1)

    def test_recover_known_global_field_from_point_and_average_observations(self):
        rng=np.random.default_rng(7);x=rng.uniform(-.3,.3,(60,3));true_centres=x[:3];coef=rng.normal(size=(3,3))
        radius=.4
        def f(p):
            k=pair_kernel((p[:,None]-true_centres).reshape(-1,3),radius,True).reshape(len(p),3,3,3)
            return np.einsum('nmij,mj->ni',k,coef)
        wall=np.array([[-1,-1,-1],[1,1,1],[-1,1,-1],[1,-1,1]],float)
        inlet=np.array([[2,2,2],[2,-2,2]],float)
        val=np.zeros(len(x),bool);val[-6:]=True
        for averaged in [False,True]:
            p,w=voxel_quadrature(x,.04,2) if averaged else (x[:,None],np.ones(1))
            clean=np.einsum('nqc,q->nc',f(p.reshape(-1,3)).reshape(p.shape),w)
            obs=Observations.make([0,10],p,w,np.stack([clean,clean*2]),val,np.arange(len(x)),0,1)
            models,audit=fit_compact_pair(obs,wall,inlet,np.zeros((2,len(inlet),3)),True,[radius],[0],
                                          max_iter=5000,tol=1e-12,polynomial=False)
            query=rng.uniform(-.2,.2,(12,3))
            np.testing.assert_allclose(models[0].predict(query),f(query),atol=3e-6)
            self.assertTrue(audit['global_solenoidal_basis'])
            self.assertFalse(audit['selection_uses_hidden_targets'])

    def test_full_basis_with_affine_background_remains_solenoidal(self):
        rng=np.random.default_rng(59);centres=rng.uniform(-.4,.4,(8,3))
        model=CompactRBF(centres,.7,True)
        model.coefficients=rng.normal(size=3*len(centres)+11)
        x=rng.uniform(-.5,.5,(10,3));errors=[]
        for eps in [1e-4,5e-5]:
            div=np.zeros(len(x))
            for c in range(3):
                d=np.eye(3)[c]*eps
                div+=(model.predict(x+d)[:,c]-model.predict(x-d)[:,c])/(2*eps)
            errors.append(np.linalg.norm(div))
        self.assertLess(errors[1],errors[0]*.27)
        self.assertLess(errors[1],1e-4)


if __name__=='__main__':unittest.main()
