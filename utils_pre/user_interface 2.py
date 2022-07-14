#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:33:27 2020
    
@author: gardar
"""

# Define a simple selection menu function
def selection_menu(message,options,**kwargs):
    
    # Options is a dictionary of indexes (numbered items) and option descriptions
    # Let's snatch it's length
    n_options = len(options)
    
    # And define a valid selection boundary, remember 'range' upper bound is exclusive
    bounds = range(1,n_options+1)
    
    # Initialise selection criterion  
    selection = 0
    
    # While selection is out of bounds
    while selection not in bounds:
        
        # Try to get a sensible input from user
        try:
            # Print a message
            print("\n")
            print(message)
            print("-"*len(message)) # Make it pretty
            
            # Display each numbered option
            for key in sorted(options.keys()):
                print(key+". " + options[key])
            
            # Get input from user
            selection = int(input("Select by entering number and hit 'RETURN': "))
            
            # If user chose a valid selection
            if selection in bounds: 
                break
            # Else if user chose an integer but it was not within bounds
            else: 
                print("\n\n ... Invalid selection... ")
                
        # If the user can't do anything right and is writing his autobiography in the console
        except ValueError:
            print("\n\n ... Select integer value from menu ... ")
    
    # Return the users choice
    return selection 

# Dead simple yes or no menu
def yes_no_menu(message): 
    
    # Define a dictionary of valid responses and what we interpret them as
    valid = {'yes': True,
             'ye' : True,
             'y'  : True,
             'no' : False,
             'n'  : False}
    
    # This explains itself...
    answer = None
    
    while answer not in valid:
        try: 
            answer = str(input(message)).lower()
            if answer in valid: 
                break
            else: 
                print("\nType 'yes' / 'no' and hit 'RETURN'")
        except ValueError:
            print("Please enter response in the form 'yes' / 'no' or 'y' / 'n'")
            
    return valid[answer]

# Print progress
def print_progress (iteration, total, message = '', length = 20):
    
    # Calculate percent complete
    percent = "{0:.1f}".format(iteration / total * 100)
    # Determine fill of loading bar, length is the total length
    fill = int(length * iteration / total)
    # Determine the empty space of the loading bar
    empty = ' ' * (length - fill)
    # Animate the bar with unicode character 2588, a filled block
    bar = u"\u2588" * fill + empty
    
    # Print loading bar
    print(f'\r{message} |{bar}| {percent}% ', end = '\r')
   
    # Print new line on completion
    if iteration == total:
        print()

# We may test our class
if __name__ == '__main__':
    
    message = "Please select an option:"
    options = {'1': 'This',
               '2': 'That'}
    
    selection = selection_menu(message,options)
    
    print("Great you've selected: {}".format(selection))
    
    answer = yes_no_menu("Select [y] or [n]: ")
    
    print('Your answer was interpreted as: {}'.format(answer))

    import time
    
    for i in range(100):
        time.sleep(0.01)
        print_progress(i+1,100, "Progress")
