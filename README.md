# cn_missings

IN dataset remarks:

- in the IntelLabs dataset, there is one row with data from april. Check if there are more from this month or if this is isolated
- time is not monotically increasing; I added code to make sure it is.
- imputation on the IN dataset results in a file of over 5 GBs... something is going wrong

TO DO:

- add an constants.py file, in which we define legend labels and other constants that could be nice to collect in a central location.
- add robust paths everywhere (e.g. eval of real missings)
- fix error in DEMS (AttributeError: 'Series' object has no attribute 'iteritems') for the iteritems on line 37
- add datetime conversion for the intellab dataset
- make CN-specific stuff (such as specifying dtypes in dataloader) general, as to accomodate for the intellabs dataset
- to save space: add remove all imputed files from git, only keep the calculated metrics
- AKE: now we run it with 5 neighbors. A different number of neighbors might improve performance.

Done: 
- Steven: fix AKE
- Steven: fix simple_eval function (probably does not work as it does not read results from disk --> change columns)