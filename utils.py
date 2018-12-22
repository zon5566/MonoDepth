#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

