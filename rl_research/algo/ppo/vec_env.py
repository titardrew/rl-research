import multiprocessing as mp
import cloudpickle
import cv2

def get_vec_environment(env_fn, n_workers):
    venv = VecEnv(env_fn, n_workers) # TODO: Add wrappers argument
    return venv


class CloudpickleWrapper(object):
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, obs):
        self.fn = cloudpickle.loads(obs)


class VecEnv:
    """
        DESCRIPTION. The class manages the separate processes which run the
            instances of an envitonment. The implementation is quite common
            in RL applications. The processes are spawned in constructor and
            the object gets the pipe handles for each process. The
            communication is very simple. The object sends a tuple of command
            and data. The command is a string with request type like "step" or
            "render". The data is usually passed to the corresponding method.
            The process sends some results back, like (obs, rew, done, info)
            tuple.

        NOTE, that we don't assume any particular structure of observations,
            rewards and information. We move most preprocessing out of the
            environment to the model implementation. Thus, we do not need to
            make the wrapper system unreasonably complex and crappy.

        RECORDING. To start recording one should call
                `start_recording(self, n_episodes, mode)`
            method. This will initiate recording from the beginning of the
            next episode for each env_worker. When the total number of
            recorded episodes is as was requested, use
                `try_get_recordings(self)`
            to get the buffer and to clean up the object recording state.
            If the recording is still in progress, the method will return None.

            _Don't put_ a postprocessing of the recording (e.g merging game
            frames into mpeg file with pyav, etc.) inside the VecEnv. This
            will make the code even more messy and unreadable, because of
            enormous cases that the code should cover. For instance, we may
            want to compress the environment's `render()` outputs, or would
            like to make split-screen recording, or to deal with complicated
            `render()` outputs.
    """

    def __init__(self, env_fn, n_workers):
        temp_env = env_fn()
        self.action_space = temp_env.action_space
        self.observation_space = temp_env.observation_space
        temp_env.close()

        self.n_workers = n_workers
        self.n_episodes = 0
        self.workers = []

        ctx = mp.get_context(
            "forkserver"
            if "forkserver" in mp.get_all_start_methods()
            else "spawn"
        )

        for idx in range(self.n_workers):
            parent_conn, worker_conn = ctx.Pipe(duplex=True)
            ps = ctx.Process(
                target=env_worker,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    CloudpickleWrapper(env_fn),
                    idx,
                ),
            )
            ps.daemon = True
            ps.start()
            self.workers.append((parent_conn.recv, parent_conn.send))

        self.recording_epi_remains = 0
        self.recording_buffer = []
        self.recording_buffer_epi = []
        self.recording_render_mode = "rgb_array"
        self.recording_in_progress = False

    def start_recording(self, n_episodes, size=None, mode="rgb_array", force_restart=False):
        if force_restart or not self.recording_in_progress:
            self.recording_in_progress = True
            self.recording_epi_remains = n_episodes
            self.recording_buffer = []
            self.recording_buffer_epi = [[] for _ in range(self.n_workers)]
            self.recording_render_mode = mode
            self.recording_size = size
            self.recording_mask = [0 for _ in range(self.n_workers)]
            return True
        return False

    def try_get_recordings(self):
        if self.recording_epi_remains == 0 and self.recording_in_progress:
            buf = self.recording_buffer
            self.recording_buffer = []
            self.recording_buffer_epi.clear()
            self.recording_mask = [0 for _ in range(self.n_workers)]
            self.recording_in_progress = False
            return buf
        return None

    def _record_all(self):
        if self.recording_epi_remains > 0:
            recs = self.render_all(mode=self.recording_render_mode,
                                   size=self.recording_size)
            for i, (rec, m) in enumerate(zip(recs, self.recording_mask)):
                if m: self.recording_buffer_epi[i].append(rec)

    def _record_idx(self, idx):
        if self.recording_epi_remains > 0 and self.recording_mask[idx]:
            rec = self.render_idx(idx, mode=self.recording_render_mode,
                                       size=self.recording_size)
            self.recording_buffer_epi[idx].append(rec)

    def step_nonstop(self, act):
        for i in range(self.n_workers):
            _, send_fn = self.workers[i]
            send_fn(("step", act[i]))

        obs, reward, done, info = [], [], [], []
        for k in range(self.n_workers):
            recv_fn, _ = self.workers[k]
            o, r, d, i = recv_fn()
            if d:
                if self.recording_epi_remains > 0:
                    if self.recording_mask[k]:
                        self.recording_epi_remains -= self.recording_mask[k]
                        rec = self.recording_buffer_epi[k]
                        self.recording_buffer.append(rec)
                        self.recording_buffer_epi[k] = []
                    else:
                        self.recording_mask[k] = 1

                self.n_episodes += 1
                o = self.reset_idx(k)
            obs.append(o)
            reward.append(r)
            done.append(d)
            info.append(i)

        self._record_all()
        return obs, reward, done, info

    def reset_all(self):
        for i in range(self.n_workers):
            _, send_fn = self.workers[i]
            send_fn(("reset", None))

        obs = []
        for i in range(self.n_workers):
            recv_fn, _ = self.workers[i]
            obs.append(recv_fn())

        self._record_all()
        return obs

    def reset_idx(self, idx):
        recv_fn, send_fn = self.workers[idx]
        send_fn(("reset", None))
        obs = recv_fn()

        self._record_idx(idx)
        return obs

    def render_all(self, mode, size=None):
        for i in range(self.n_workers):
            _, send_fn = self.workers[i]
            send_fn(("render", (mode, size)))

        recs = []
        for i in range(self.n_workers):
            recv_fn, _ = self.workers[i]
            recs.append(recv_fn())
        return recs

    def render_idx(self, idx, mode, size=None):
        recv_fn, send_fn = self.workers[idx]
        send_fn(("render", (mode, size)))
        rec = recv_fn()
        return rec

    def close(self):
        for i in range(self.n_workers):
            _, send_fn = self.workers[i]
            send_fn(("close", None))

    def seed(self, seed):
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.n_workers)]
        elif not isinstance(seed, list):
            raise ValueError("Seed must be either int or list of ints")

        for i in range(self.n_workers):
            _, send_fn = self.workers[i]
            send_fn(("seed", seed[i]))
            self.action_space.np_random.seed(seed[i])

        seed_hist = []
        for i in range(self.n_workers):
            recv_fn, _ = self.workers[i]
            seed_hist.append(recv_fn())

        return seed_hist

    def random_action(self):
        act = []
        for i in range(self.n_workers):
            act.append(self.action_space.sample())
        return act


def env_worker(recv_fn, send_fn, env_fn, idx):
    env = env_fn.fn()
    while True:
        try:
            request, data = recv_fn()
            if request == "reset":
                obs = env.reset()
                send_fn(obs)
            elif request == "step":
                act = data
                obs, reward, done, info = env.step(act)
                send_fn((obs, reward, done, info))
            elif request == "render":
                mode, size = data
                rec = env.render(mode=mode)
                if size:
                    if isinstance(size, int):
                        w = int(size * rec.shape[1] / rec.shape[0])
                        h = size
                    elif isinstance(size, tuple) or isinstance(size, list):
                        w, h = size
                    rec = cv2.resize(rec, (w, h))
                send_fn(rec)
            elif request == "seed":
                seed = data
                seeds = env.seed(seed)
                send_fn(seeds)
            elif request == "close":
                env.close()
                return
            else:
                send_fn((None))
        except EOFError:
            break
