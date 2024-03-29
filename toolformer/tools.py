# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_tools.ipynb.

# %% auto 0
__all__ = ['Calendar', 'Calculator']

# %% ../nbs/03_tools.ipynb 2
import requests
import calendar
import datetime
import os

# %% ../nbs/03_tools.ipynb 3
def Calendar():
    now = datetime.datetime.now()
    return f'Today is {calendar.day_name[now.weekday()]}, {calendar.month_name[now.month]} {now.day}, {now.year}.'

# %% ../nbs/03_tools.ipynb 5
def Calculator(expression:str): 
    expression = expression.replace('x', '*')
    return eval(expression)
