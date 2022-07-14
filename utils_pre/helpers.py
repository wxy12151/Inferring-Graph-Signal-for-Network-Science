#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:56:31 2021

Utility library for helper functions

@author: gardar
"""


def list_of_dict_search(search_item, search_key, list_of_dicts):
    '''
    Helper function for searching within a list of dictionaries
    Returns boolean 'True' if a hit is found, 'False' otherwise
    
    Parameters
    ----------
    search_item : Item to search for
    search_key : Which key to query 
    list_of_dicts : The list of dictionaries to query
    
    Returns
    -------
    wasFound : Boolean True if hit, else False
    '''    
    
    # Initialise the hit value as false
    wasFound = False
    
    # For each dictionary within the list
    for dictionary in list_of_dicts:
        
        # Check if the value assigned to the search key matches the search item
        if dictionary[search_key] == search_item:
        
            # If so, we have a hit...
            wasFound = True
            
            # ... and can stop iteration
            break
        
        # If not
        else:
            
            # We continue searching
            continue
    
    # Return the hit status
    return wasFound

if __name__=="__main__":
        
    # List of dictionary search sanity check
    l_of_d = [{'id':1,
               'x' : -215.6,
               'y' : 513.2}
              , 
              {'id': 97,
               'x' : 21.9,
               'y' : 78.9}]
    
    item = 97
    key  = 'id'
    wasFound = list_of_dict_search(search_item = item, search_key = key, list_of_dicts=l_of_d) 
    
    # Print search result
    print("The value: '{search}' {result} found".format(search=item, 
                                                        result=('was' if wasFound else 'was NOT')))