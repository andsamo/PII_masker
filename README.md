# PII_masker

## Description
Preprocess text data by masking personally identifying information (PII) to pseudo-anonymize, uses a SpaCy nlp model

## Features
- masks PII including names, companies, locations, job titles, pronouns, etc.
- Handles text normalization to clean irregular formatting
- Manages Excel cell length limits automatically
- Provides progress tracking for large datasets

## How to use
> # initialize model
> masker = PIIMasker('en_core_web_trf') # specifies transformer model

> # process all text columns
> df_anonymized = masker.process_dataframe(df)

> # or you can process specific columns
> df_anonymized = masker.process_dataframe(df, text_columns=['ColA', 'ColB'])

