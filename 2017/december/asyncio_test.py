import asyncio
import time

"""
asyncio  python自带的异步框架。
协程：是一种运行在用户态的轻量级类线程。
之所以说是类线程，是因为协程并不会单独的运行在其线程中，而是多个协程运行在同一个线程之中，
从而节省了上下文切换开销，但是也意味着协程只适合完成轻量级的任务。
"""

now = lambda: time.time()


async def do_some_work(x):
    print("wating: ", x)


def simple_demo():
    start = now()

    coroutine = do_some_work(1)
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(coroutine)
    print('Time: ', now() - start)


def task_demo():
    start = now()

    coroutine = do_some_work(2)
    task = asyncio.ensure_future(coroutine)
    loop = asyncio.get_event_loop()
    print(task)
    loop.run_until_complete(task)
    print(task)
    print('Time: ', now() - start)


async def do_some_work_1(x):
    print('Waiting: ', x)
    return 'Done after {}s'.format(x)


def callback_demo():
    def callback(future):
        print("Callback: ", future.result())

    start = now()
    coroutine = do_some_work_1(3)
    loop = asyncio.get_event_loop()
    task = asyncio.ensure_future(coroutine)
    task.add_done_callback(callback)
    loop.run_until_complete(task)
    print('Time: ', now() - start)


async def do_some_work_3(x):
    print("Waiting: ", x)
    await asyncio.sleep(x)  # asyncio.sleep()返回了一个协程，在指定时间后完成
    # await 挂起当前协程(do_some_work_3)，然后将之后的一个协程交给事件循环（怎么获取到事件循环的？）
    return 'Done after {} s'.format(x)


def await_demo():
    start = now()
    coroutines = [do_some_work_3(x) for x in [1, 2, 4]]
    tasks = [asyncio.ensure_future(co) for co in coroutines]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))  # asyncio.wait接受一个task列表

    [print('RET: ', task.result()) for task in tasks]
    print('Time: ', now() - start)
