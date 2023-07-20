# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_datasets.ipynb.

# %% ../nbs/04_datasets.ipynb 2
from __future__ import annotations
import math, random, torch, matplotlib.pyplot as plt, numpy as np, matplotlib as mpl, shutil, os, gzip, pickle, re, copy, time
from pathlib import Path
from functools import partial
import fastcore.all as fc
from glob import glob

from torch import tensor, nn, optim
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, default_collate
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence

from datetime import datetime, timedelta
import calendar
import json
import csv

# %% auto 0
__all__ = ['CALENDAR_PROMPT', 'DATESET_HOLIDAYS', 'DATESET_TEMPLATES', 'CALCULATOR_PROMPT', 'random_date', 'format_date',
           'get_prompt', 'make_dateset', 'PromptDS']

# %% ../nbs/04_datasets.ipynb 4
CALENDAR_PROMPT = """Your task is to add calls to a Calendar API to a piece of text. 
The API calls should help you get information required to complete the text. 
You can call the API by writing <% Calendar() %>. 
Here are some examples of API calls:
Input: Today is the first Friday of the year.
Output: Today is the first <% Calendar() %> Friday of the year.
Input: The president of the United States is Joe Biden.
Output: The president of the United States is <% Calendar() %> Joe Biden.
Input: The current day of the week is Wednesday.
Output: The current day of the week is <% Calendar() %> Wednesday.
Input: The number of days from now until Christmas is 30.
Output: The number of days from now until Christmas is <% Calendar() %> 30.
Input: The store is never open on the weekend, so today it is closed.
Output: The store is never open on the weekend, so today <% Calendar() %> it is closed.
Input: [INPUT]
Output: """

# %% ../nbs/04_datasets.ipynb 5
DATESET_HOLIDAYS = [
    "New Year's Day",
    "The Birthday of Martin Luther King, Jr.",
    "Washington's Birthday",
    "Memorial Day",
    "Juneteenth National Independence Day",
    "Independence Day",
    "Labor Day",
    "Columbus Day",
    "Veterans Day",
    "Thanksgiving Day",
    "Christmas Day"
]

# %% ../nbs/04_datasets.ipynb 6
DATESET_TEMPLATES = [
    {
        "main": "How many days {var_1} {var_2}?",
        "vars": [['ago was', 'are there until'], ['past_date', 'future_date']],
        "p": 0.05,
        "logic": None,
        "holiday": False
    },
    {
        "main": "What {var_1} was it {logic} days ago?",
        "vars": [['day of the week', 'day of the month', 'month', 'year']],
        "p": 0.08,
        "logic": 'current_date – past_date',
        "holiday": False
    },
    {
        "main": "What {var_1} will it be in {logic} days?",
        "vars": [['day of the week', 'day of the month', 'month', 'year']], 
        "p": 0.08,
        "logic": 'future_date – current_date',
        "holiday": False
    },
    {
        "main": "What day of the week {var_1} it on {var_2}?",
        "vars": [['is', 'was'], ['future_date', 'past_date']],
        "p": 0.05,
        "logic": None,
        "holiday": False
    },
    {
        "main": "What {var_1} is it {var_2}?",
        "vars": [['day of the week', 'day of the month', 'month', 'year'], ['the day before yesterday', 'today', 'tomorrow', 'the day after tomorrow']],
        "p": 0.42,
        "logic": None,
        "holiday": False
    },
    {
        "main": "What {var_1} is {holiday} this year?",
        "vars": [['day of the week', 'day of the month', 'month']],
        "p": 0.2,
        "logic": None,
        "holiday": True
    },
    {
        "main": "How many {var_1} {holiday} this year?",
        "vars": [['days ago was', 'weeks ago was', 'months ago was', 'years ago was', 'days are there until', 'weeks are there until', 'months are there until', 'years are there until']],
        "p": 0.12,
        "logic": None,
        "holiday": True
    }
]

# %% ../nbs/04_datasets.ipynb 7
def random_date(start, end): return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# %% ../nbs/04_datasets.ipynb 8
def format_date(date): return f'{calendar.day_name[date.weekday()]}, {calendar.month_name[date.month]} {date.day}, {date.year}'

# %% ../nbs/04_datasets.ipynb 9
def get_prompt(template, holidays):
    current_date = datetime.now()
    past_date = random_date(datetime(1750, 1, 1), current_date)
    future_date = random_date(current_date, datetime(2200, 1, 1))
    
    prompt = template['main']
    
    # replace vars
    # import pdb; pdb.set_trace()
    r = random.randint(0,len(template['vars'][0])-1)
    for i, var in enumerate(template['vars']):
        v = var[r]
        if v == 'future_date': v = format_date(future_date)
        elif v == 'past_date': v = format_date(past_date)
        elif v == 'current_date': v = format_date(current_date)
        prompt = prompt.replace(f'{{var_{i + 1}}}', v)
    
    # do any logic
    if template['logic'] is not None:
        if template['logic'] == 'future_date – current_date': 
            res = (future_date - current_date).days
            prompt = prompt.replace(f'{{logic}}', str(res))
        if template['logic'] == 'current_date – past_date':
            res = (current_date - past_date).days
            prompt = prompt.replace(f'{{logic}}', str(res))
            
    # get holiday if needed
    if template['holiday'] is not None:
        r = random.randint(0,len(holidays)-1)
        h = holidays[r]
        prompt = prompt.replace('{holiday}', h)
        
    return prompt

# %% ../nbs/04_datasets.ipynb 10
def make_dateset(templates, holidays, size=9400):
    prompts = []
    for t in templates:
        num = int(t['p']*size)
        for _ in range(num):
            p = get_prompt(t, holidays)
            prompts.append(p)
    return prompts

# %% ../nbs/04_datasets.ipynb 14
CALCULATOR_PROMPT = f"""Your task is to add calls to a Calculator API to a piece of text. 
The calls should help you get information required to complete the text. 
You can call the API by writing "<% Calculator(expression) %>" where "expression" is the expression to be computed.
You should simply return the same text with the API call included.
Here are some examples of API calls: 
Input: The number in the next term is 18 + 12 x 3 = 54. 
Output: The number in the next term is 18 + 12 x 3 = <% Calculator(18 + 12 * 3) %> 54. 
Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people. 
Output: The population is 658,893 people. This is 11.4% of the national average of <% Calculator(658,893 / 11.4) %> 5,763,868 people. 
Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year. 
Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of <% Calculator(723 / 252) %> 2.87 per match). This is twenty goals more than the <% Calculator(723 - 20) %> 703 goals last year. 
Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years. 
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was <% Calculator(2011 - 1994) %> 17 years. 
Input: From this, we have 4 * 30 minutes = 120 minutes. 
Output: From this, we have 4 * 30 minutes = <% Calculator(4 * 30) %> 120 minutes. 
Input: [INPUT] 
Output: """

# %% ../nbs/04_datasets.ipynb 22
class PromptDS:
    """
        Returns a tuple containing the whole prompt (including the in-context
        teacher prefix), the varying input data sequence and the start index
        of the model's response.
    """
    def __init__(self, data): fc.store_attr()
    def __len__(self): return len(self.data)
    def __getitem__(self, i): 
        prompt, inp = self.data[i]
        prompt = prompt.replace("[INPUT]", inp)
        return prompt, (inp, len(prompt))
