# DA150X
Kex jobb, inom quantum computing


errors
File c:\venv\Lib\site-packages\qiskit\utils\parallel.py:189, in parallel_map(task, values, task_args, task_kwargs, num_processes)
    187 results = []
    188 for _, value in enumerate(values):
--> 189     result = task(value, *task_args, **task_kwargs)
    190     results.append(result)
...
--> 211     raise LinAlgError("Schur form not found. Possibly ill-conditioned.")
    213 if sort is None:
    214     return result[0], result[-3]

LinAlgError: Schur form not found. Possibly ill-conditioned

QiskitError: 'TwoQubitWeylDecomposition: failed to diagonalize M2. Please report this at https://github.com/Qiskit/qiskit-terra/issues/4159. Input: [[Complex { re: 0.19683248581778373, im: 0.19824030227267975 }, Complex { re: -0.20419798171402923, im: -0.19015072286758328 }, Complex { re: 0.1853098388249213, im: 0.6230741577443535 }, Complex { re: -0.6279329389464411, im: -0.16505369947567125 }],\n [Complex { re: -0.19659509651192095, im: -0.1980012150734224 }, Complex { re: -0.20444455153188937, im: -0.1903803305684238 }, Complex { re: -0.1850863463774317, im: -0.6223227007823545 }, Complex { re: -0.6286911702914875, im: -0.16525300242789018 }],\n [Complex { re: -0.46039439162955287, im: 0.45763392944166437 }, Complex { re: 0.4723909181975835, im: -0.4467198821661805 }, Complex { re: 0.26735367100811885, im: -0.07967620067124363 }, Complex { re: -0.08804359215082865, im: 0.2651761708519537 }],\n [Complex { re: 0.4616247703758612, im: -0.457836656564122 }, Complex { re: 0.4721488769361613, im: -0.4454966298730346 }, Complex { re: -0.26786074042393554, im: 0.07950282665461648 }, Complex { re: -0.08820063195500578, im: 0.2646638072683473 }]], shape=[4, 4], strides=[4, 1], layout=Cc (0x5), const ndim=2'