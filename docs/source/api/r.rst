R API
=====

Part of :doc:`index` — MORIE API reference.

Reference for every public function exported by the
``morie`` R package. Signatures and descriptions come from the
Roxygen2 ``.Rd`` files in ``r-package/morie/man/``; see
:doc:`../methods/index` for the methodology behind each function.

Causal estimators
-----------------

.. r:function:: estimate_aipw
.. r:function:: estimate_atc
.. r:function:: estimate_ate
.. r:function:: estimate_att
.. r:function:: estimate_cate
.. r:function:: estimate_g_computation
.. r:function:: estimate_gate
.. r:function:: estimate_late
.. r:function:: estimate_propensity_scores

Effect sizes + tests
--------------------

.. r:function:: anova_one_way
.. r:function:: chi_square_test
.. r:function:: cohens_d
.. r:function:: cramers_v
.. r:function:: e_value
.. r:function:: effective_sample_size
.. r:function:: eta_squared
.. r:function:: fisher_exact_test
.. r:function:: hedges_g
.. r:function:: kendall_tau
.. r:function:: kruskal_wallis_test
.. r:function:: levene_test
.. r:function:: mann_whitney_test
.. r:function:: odds_ratio_ci
.. r:function:: omega_squared
.. r:function:: one_sample_t_test

Survey + sampling
-----------------

.. r:function:: bootstrap_sample
.. r:function:: calibration_weights
.. r:function:: cluster_sample
.. r:function:: compute_design_weights
.. r:function:: design_effect
.. r:function:: generate_synthetic_data
.. r:function:: jackknife_estimate

Datasets + I/O
--------------

.. r:function:: canonicalize_cpads_data
.. r:function:: load_cpads_data
.. r:function:: morie_assistant_query
.. r:function:: morie_builtin_db
.. r:function:: morie_cache_file
.. r:function:: morie_cache_list
.. r:function:: morie_cache_load
.. r:function:: morie_cache_store
.. r:function:: morie_dataset_catalog
.. r:function:: morie_dataset_info
.. r:function:: morie_db_connect
.. r:function:: morie_download_bootstrap
.. r:function:: morie_fetch_ckan
.. r:function:: morie_list_datasets
.. r:function:: morie_load_cpads
.. r:function:: morie_load_dataset
.. r:function:: morie_paths
.. r:function:: morie_userguide

Workflow + audit
----------------

.. r:function:: ask_percy
.. r:function:: audit_public_outputs
.. r:function:: build_assistant_prompt
.. r:function:: build_outputs_manifest
.. r:function:: build_prompt
.. r:function:: cpads_contract
.. r:function:: default_synthetic_name_map
.. r:function:: default_workflow_map
.. r:function:: find_project_root
.. r:function:: list_morie_modules

Other
-----

.. r:function:: paired_t_test
.. r:function:: point_biserial_r
.. r:function:: power_prop_test
.. r:function:: power_t_test
.. r:function:: pps_sample
.. r:function:: proportion_ci
.. r:function:: read_outputs_manifest
.. r:function:: risk_difference_ci
.. r:function:: risk_ratio_ci
.. r:function:: run_ebac_selection_ipw_analysis
.. r:function:: run_morie_module
.. r:function:: run_morie_modules
.. r:function:: run_pipeline
.. r:function:: run_propensity_ipw_analysis
.. r:function:: run_workflow_step
.. r:function:: sample_size_logistic
.. r:function:: sensitivity_rosenbaum
.. r:function:: shapiro_wilk_test
.. r:function:: simple_random_sample
.. r:function:: spearman_rho
.. r:function:: stratified_sample
.. r:function:: summarize_output_audit
.. r:function:: two_sample_t_test
.. r:function:: validate_cpads_data
.. r:function:: validate_outputs_manifest
.. r:function:: wilcoxon_signed_rank_test
.. r:function:: write_synthetic_data

