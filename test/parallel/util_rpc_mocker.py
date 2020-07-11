from multiprocessing import Manager
from machin.parallel.thread import Thread
from queue import Empty
import dill
import time
import random
import threading

rpc_mocker_lock = None
rpc_mocker_inited = False
rpc_mocker_running = False
rpc_mocker_process_id = None
rpc_mocker_cmd_conn = None
rpc_mocker_timeout = None
rpc_mocker_workers = []


def send(cmd_conn, obj, side):
    # send and recv must be paired on two sides!
    cmd_conn[0]["data"] = obj
    if side == 0:
        cmd_conn[1].set()
    else:
        cmd_conn[2].set()


def recv(cmd_conn, side):
    # send and recv must be paired on two sides!
    if side == 0:
        cmd_conn[2].wait()
        cmd_conn[2].clear()
        return cmd_conn[0]["data"]
    else:
        cmd_conn[1].wait()
        cmd_conn[1].clear()
        return cmd_conn[0]["data"]


def poll(cmd_conn, side):
    if side == 0:
        return cmd_conn[2].is_set()
    else:
        return cmd_conn[1].is_set()


def make_conn(manager):
    return manager.dict(), manager.Event(), manager.Event()


class RpcMocker(object):
    def __init__(self, process_num, rpc_worker_num=4,
                 rpc_init_wait_time=1, rpc_timeout=1,
                 rpc_response_time=(1e-3, 1e-1), result_expire_time=1,
                 print_expired=True):
        self.process_num = process_num
        self.rpc_worker_num = rpc_worker_num
        self.rpc_init_wait_time = rpc_init_wait_time
        self.rpc_timeout = rpc_timeout
        self.rpc_response_time = rpc_response_time
        self.result_expire_time = result_expire_time
        self.print_expired = print_expired

        self._ctx_man = Manager()
        # Pipe is the most reasonable choice, but could not be used here:
        # eg: suppose we are using "spawn" instead of "fork" to start
        # new processes
        # in process A (eg: pytest fixture) we create the mocker
        # then in process B we perform RpcMocker.start() and
        # spawn multiple process Cs,
        # and in C we perform injected init_rpc(),
        # because A have exited, the pipes are closed and therefore in
        # B and C we have invalid pipe file descriptors.
        # We use managed resources so things can be used and cleaned up safely.

        # two events are used to notify two ends
        self._cmd_conn = [make_conn(self._ctx_man)
                          for _ in range(process_num)]
        self._delay_queue = self._ctx_man.Queue()
        self._cmd_queues = [self._ctx_man.Queue() for _ in range(process_num)]

        self._stop = False
        self._route_table = {}
        self._result_write_lock = threading.Lock()
        self._result_table = {}
        self._request_counter = 0
        self._router = Thread(target=self._router_main)
        self._result_maintainer = Thread(
            target=self._result_maintainer_main
        )

    def start(self):
        self._router.start()
        self._result_maintainer.start()

    def watch(self):
        self._router.watch()
        self._result_maintainer.watch()

    def stop(self):
        self._stop = True
        self._router.join()
        self._result_maintainer.join()

    @staticmethod
    def interposer(obj, timestamp, src, to, fuzz, token):
        # can be overloaded
        # controls request delivery, response is not controlled
        if fuzz["drop"]:
            return "drop", obj, timestamp, src, to, fuzz, token
        if fuzz["delay"]:
            if time.time() - timestamp < fuzz["delay_time"]:
                return "delay", obj, timestamp, src, to, fuzz, token
        return "ok", obj, timestamp, src, to, fuzz, token

    @staticmethod
    def route_fuzzer():
        # can be overloaded
        return {"drop": False, "delay": False, "delay_time": 0}

    def get_mocker_init_rpc(self, process_id):
        # get a mocker for rpc.init_rpc, which will be executed in a subprocess
        # requires dill to dump and load the returned function
        cmd_conn = self._cmd_conn[process_id]
        worker_num = self.rpc_worker_num
        timeout = self.rpc_timeout
        wait_time = self.rpc_init_wait_time
        queue = self._cmd_queues[process_id]

        def init_rpc(name, _backend=None, _rank=None, _world_size=None,
                     rpc_backend_options=None):
            nonlocal worker_num
            global rpc_mocker_lock, \
                rpc_mocker_inited, \
                rpc_mocker_running, \
                rpc_mocker_process_id, \
                rpc_mocker_cmd_conn, \
                rpc_mocker_timeout, \
                rpc_mocker_workers

            if rpc_mocker_inited:
                raise RuntimeError("Rpc already initialized!")

            rpc_mocker_lock = threading.Lock()
            rpc_mocker_running = True
            rpc_mocker_process_id = process_id
            rpc_mocker_cmd_conn = cmd_conn
            rpc_mocker_timeout = timeout
            if rpc_backend_options is not None:
                worker_num = rpc_backend_options.num_send_recv_threads
                rpc_mocker_timeout = \
                    rpc_backend_options.rpc_timeout.total_seconds()

            rpc_mocker_workers = [threading.Thread(
                target=RpcMocker._rpc_worker, args=(queue,)
            ) for _ in range(worker_num)]
            for worker in rpc_mocker_workers:
                worker.daemon = True
                worker.start()

            with rpc_mocker_lock:
                send(cmd_conn, ("init_rpc", name), 0)
                if not recv(cmd_conn, 0):
                    raise RuntimeError("Duplicate rpc name: {}".format(name))
            time.sleep(wait_time)
            rpc_mocker_inited = True

        return init_rpc

    @staticmethod
    def get_mocker_shutdown():
        def shutdown(graceful=True):
            global rpc_mocker_inited, \
                rpc_mocker_running, \
                rpc_mocker_workers

            if not rpc_mocker_inited:
                raise RuntimeError("Rpc not initialized!")
            rpc_mocker_running = False
            for worker in rpc_mocker_workers:
                worker.join()

        return shutdown

    def get_mocker_rpc_async(self):
        report_delay = random.random() * \
                       (self.rpc_response_time[1] -
                        self.rpc_response_time[0]) + \
                       self.rpc_response_time[0]

        def rpc_async(to, func, args=(), kwargs={}, timeout=-1.0):
            global rpc_mocker_lock, \
                rpc_mocker_inited, \
                rpc_mocker_cmd_conn, \
                rpc_mocker_timeout

            if not rpc_mocker_inited:
                raise RuntimeError("Rpc not initialized!")

            cmd_conn = rpc_mocker_cmd_conn
            with rpc_mocker_lock:
                send(cmd_conn, ("rpc_async", to, func, args, kwargs), 0)
                status, token = recv(cmd_conn, 0)

            if status == "unresolved":
                raise RuntimeError("Rpc to {} failed".format(to))
            elif status == "drop":
                # async will not raise errors
                pass

            if not timeout > 0:
                timeout = rpc_mocker_timeout

            # delay or ok
            timestamp = time.time()

            class Waiter:
                @staticmethod
                def wait():
                    nonlocal status, to
                    if status == "drop":
                        raise RuntimeError("Rpc to {} failed".format(to))
                    while True:
                        with rpc_mocker_lock:
                            send(cmd_conn, ("rpc_async_result_poll", token), 0)
                            if recv(cmd_conn, 0):
                                break
                        if time.time() - timestamp >= timeout:
                            raise TimeoutError("Rpc timeout")
                        time.sleep(1e-3)
                    with rpc_mocker_lock:
                        send(cmd_conn, ("rpc_async_result", token), 0)
                        status, result = recv(cmd_conn, 0)
                        if status == "expired":
                            raise RuntimeError("Result expired")
                    time.sleep(report_delay)
                    return result

            return Waiter()

        return rpc_async

    def get_mocker_rpc_sync(self):
        report_delay = random.random() * \
                       (self.rpc_response_time[1] -
                        self.rpc_response_time[0]) + \
                       self.rpc_response_time[0]

        def rpc_sync(to, func, args=(), kwargs={}, timeout=-1.0):
            global rpc_mocker_lock, \
                rpc_mocker_inited, \
                rpc_mocker_cmd_conn, \
                rpc_mocker_timeout

            if not rpc_mocker_inited:
                raise RuntimeError("Rpc not initialized!")

            cmd_conn = rpc_mocker_cmd_conn
            if not timeout > 0:
                timeout = rpc_mocker_timeout
            with rpc_mocker_lock:
                send(cmd_conn, ("rpc_sync", to, func, args, kwargs), 0)
                status, token = recv(cmd_conn, 0)

            if status == "unresolved":
                raise RuntimeError("Rpc to {} failed".format(to))
            elif status == "drop":
                raise RuntimeError("Rpc dropped")

            if not timeout > 0:
                timeout = rpc_mocker_timeout

            # delay or ok
            timestamp = time.time()
            while True:
                with rpc_mocker_lock:
                    send(cmd_conn, ("rpc_sync_result_poll", token), 0)
                    if recv(cmd_conn, 0):
                        break
                if time.time() - timestamp >= timeout:
                    raise TimeoutError("Rpc timeout")
                time.sleep(1e-3)
            with rpc_mocker_lock:
                send(cmd_conn, ("rpc_sync_result", token), 0)
                status, result = recv(cmd_conn, 0)
                if status == "expired":
                    raise RuntimeError("Result expired")
            time.sleep(report_delay)
            return result

        return rpc_sync

    def _result_maintainer_main(self):
        while not self._stop:
            expired_tokens = []
            if self._result_write_lock.acquire(timeout=1e-3):
                for k, v in self._result_table.items():
                    if (v is not None and
                            time.time() - v[0] > self.result_expire_time):
                        expired_tokens.append(k)

                for k in expired_tokens:
                    if self.print_expired:
                        print("Rpc result expired: {}, value: {}"
                              .format(k, self._result_table[k][1]))
                    self._result_table.pop(k)
                self._result_write_lock.release()

            # sleep to prevent starve the router thread
            # and overload the lock, since it is repeatedly
            # releasing and locking, will make pytest hang
            time.sleep(self.result_expire_time / 100)

    def _router_main(self):
        while not self._stop:
            for conn, process_id in zip(self._cmd_conn,
                                        range(self.process_num)):
                if not poll(conn, 1):
                    continue
                else:
                    conn_cmd, *obj = recv(conn, 1)
                    if conn_cmd == "init_rpc":
                        name = obj[0]
                        if name in self._route_table:
                            send(conn, False, 1)
                        self._route_table[name] = process_id
                        send(conn, True, 1)
                    elif conn_cmd in {"rpc_async", "rpc_sync"}:
                        if obj[0] not in self._route_table:
                            print("Warning: {} not registered in rpc."
                                  .format(obj[0]))
                            send(conn, ("unresolved", None), 1)
                            continue

                        timestamp = time.time()
                        src = process_id
                        to = self._route_table[obj[0]]
                        fuzz = self.route_fuzzer()
                        token = self._request_counter

                        interposer_cmd, *packet = self.interposer(
                            obj, timestamp, src, to, fuzz, token
                        )

                        if interposer_cmd == "drop":
                            send(conn, ("drop", None), 1)
                            continue
                        elif interposer_cmd == "delay":
                            self._delay_queue.put_nowait(packet)
                            send(conn, ("delay", token), 1)
                            self._request_counter += 1
                        elif interposer_cmd == "ok":
                            self._cmd_queues[to].put_nowait(packet)
                            send(conn, ("ok", token), 1)
                            self._request_counter += 1
                        else:
                            raise RuntimeError("Unknown interposer command: {}"
                                               .format(interposer_cmd))
                    elif conn_cmd in {"rpc_async_result", "rpc_sync_result"}:
                        token = obj[0]
                        if self._result_write_lock.acquire(timeout=1e-3):
                            if token in self._result_table:
                                send(conn,
                                     ("ok", self._result_table[token][1]), 1)
                            else:
                                send(conn, ("expired", None), 1)
                            self._result_table.pop(token)
                            self._result_write_lock.release()
                    elif conn_cmd in {"rpc_async_result_poll",
                                      "rpc_sync_result_poll"}:
                        token = obj[0]
                        if self._result_write_lock.acquire(timeout=1e-3):
                            send(conn, token in self._result_table, 1)
                            self._result_write_lock.release()
                    elif conn_cmd == "rpc_async_sync_response":
                        to, token, result = obj
                        if self._result_write_lock.acquire(timeout=1e-3):
                            self._result_table[token] = (time.time(), result)
                            self._result_write_lock.release()
                        send(conn, True, 1)

            if not self._delay_queue.empty():
                packet = self._delay_queue.get(timeout=1e-3)
                interposer_cmd, *packet = self.interposer(*packet)
                if interposer_cmd in "delay":
                    self._delay_queue.put_nowait(packet)
                elif interposer_cmd == "ok":
                    to = packet[3]
                    self._cmd_queues[to].put_nowait(packet)
                else:
                    raise RuntimeError("Unknown interposer command: {}"
                                       .format(interposer_cmd))

    @staticmethod
    def _pack_packet(packet):
        return dill.dumps(packet)

    @staticmethod
    def _unpack_packet(raw_packet):
        return dill.loads(raw_packet)

    @staticmethod
    def _rpc_worker(queue):
        global rpc_mocker_lock, \
            rpc_mocker_running, \
            rpc_mocker_cmd_conn
        while rpc_mocker_running:
            try:
                obj, _timestamp, src, _to, _fuzz, token = \
                    queue.get(timeout=1e-3)
                _to_name, func, args, kwargs = obj
                result = func(*args, **kwargs)
                cmd_conn = rpc_mocker_cmd_conn
                with rpc_mocker_lock:
                    send(cmd_conn, ("rpc_async_sync_response",
                                    src, token, result), 0)
                    recv(cmd_conn, 0)
            except Empty:
                continue
