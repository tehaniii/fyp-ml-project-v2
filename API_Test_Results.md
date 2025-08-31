# API Test Results — 2025-08-24 13:13:21

| ID | Test Case | Description | Expected Outcome | Testing Outcome | Status |
|----|-----------|-------------|------------------|-----------------|--------|
| API1 | Pricing – Happy Path | Mocked Browse returns valid items | stats.median present, n_used ≥ 1, sample ≤ 8 | Pass | Pass |
| API2 | Pricing – No Results | Mocked Browse returns 0 items | stats.n = 0, n_used = 0, sample empty | Pass | Pass |
| API3 | Year Filter | Allow year ±1; exclude far years | Only 1808/1809 in sample | Pass | Pass |
| API4 | Forbid Filter | Exclude pendants/replicas | Forbidden tokens excluded | Pass | Pass |
| API5 | Synonyms Expansion | Heuristic denom variants tried | At least one variant yields items | Pass | Pass |
| API6 | Marketplace Switch | GB marketplace env | Result structure intact | Pass | Pass |
| API7 | Finding API – Happy Path | Exception | List result | Fail (cannot import name 'ebay_api' from 'src' (unknown location)) | Fail |
| API8 | OAuth / Network Error Handling | Simulated failure from browse_search | Exception is caught by caller (no app crash) | Pass | Pass |