import pandas as pd


def prepare_dataframe(full_df, llm_summary, lsa_summary):
    """
        Prepares clean summary DataFrames by aligning them with the original dataset and
        removing inconsistent or missing indices across LLM and LSA summaries.

        Parameters:
        -----------
        full_df : pd.DataFrame
            The original full dataset containing text and labels.
        llm_summary : dict
            Dictionary of LLM-generated summaries indexed by original row index (as string).
        lsa_summary : dict
            Dictionary of LSA-generated summaries indexed by original row index (as string).

        Returns:
        --------
        full_df : pd.DataFrame
            Filtered full dataset with rows matching LLM summary indices.
        summary_df_llm : pd.DataFrame
            DataFrame of LLM summaries with corresponding labels.
        summary_df_lsa : pd.DataFrame
            DataFrame of LSA summaries with corresponding labels.
    """
    
    # Convert dict keys (which are strings) to integers
    dict_keys_as_int = set(map(int, llm_summary.keys()))

    # Get all indices from the DataFrame
    df_indices = set(full_df.index)

    # Find indices that are in the DataFrame but NOT in the dictionary
    missing_indices = df_indices - dict_keys_as_int

    # Drop those rows
    full_df = full_df.drop(missing_indices)
    missing_indices_str = list(map(str, missing_indices))
    for idx in missing_indices_str:
        llm_summary.pop(str(idx), None)
        lsa_summary.pop(str(idx), None)

    summary_df_llm = pd.DataFrame.from_dict(llm_summary, orient='index', columns=['summary'])
    summary_df_llm.index = summary_df_llm.index.astype(int)  # convert string indices to int

    # Match and assign labels from main dataset
    summary_df_llm['label'] = full_df.loc[summary_df_llm.index, 'label'].values

    #summary_df_llm = summary_df_llm.reset_index()
    #full_df_subset = full_df[['label']].reset_index()
    #summary_df_llm = summary_df_llm.merge(full_df_subset, on='index', how='left')

    summary_df_lsa = pd.DataFrame.from_dict(lsa_summary, orient='index', columns=['summary'])
    summary_df_lsa.index = summary_df_lsa.index.astype(int)  # convert string indices to int

    # Match and assign labels from main dataset
    summary_df_lsa['label'] = full_df.loc[summary_df_lsa.index, 'label'].values

    #summary_df_lsa = summary_df_lsa.reset_index()
    #full_df_subset = full_df[['label']].reset_index()
    #summary_df_lsa = summary_df_lsa.merge(full_df_subset, on='index', how='left')

    return full_df, summary_df_llm, summary_df_lsa



