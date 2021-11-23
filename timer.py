from collections import defaultdict
from time import time

import torch


class Timer:

    TIMER = defaultdict(lambda: 0)
    COUNTER = defaultdict(lambda: 0)
    ACTIVE = True

    @staticmethod
    def start():
        if Timer.ACTIVE:
            torch.cuda.synchronize()
            s = time()
            return s

    @staticmethod
    def end(starttime, tag, is_print=False, text="", level="0"):
        if Timer.ACTIVE:
            tag = f"{level}-{tag}"
            torch.cuda.synchronize()
            end = time()
            diff = end - starttime

            if isinstance(is_print, bool) and is_print == True:
                print(f"Tag : {tag} Text : {text} Time : {diff}")

            Timer.TIMER[tag] += diff
            Timer.COUNTER[tag] += 1

    @staticmethod
    def print_all():
        TimerOutput = {}
        if Timer.ACTIVE:
            print(f"\n-----Timer Results-----\n")

            keys = sorted(Timer.TIMER.keys())
            for k in keys:
                v = Timer.TIMER[k]
                call_val = Timer.COUNTER[k]
                sublevels = len(k.split("-")[0].split(".")) - 1
                tabs = sublevels * "\t"
                print(f"{tabs} {k}  Time : {v}  Call : {call_val}  PS : {round(1/(v/call_val),2)}")
                TimerOutput[k] = {"Time":v,
                                 "PS":round(1/(v/call_val),2)}
        return TimerOutput