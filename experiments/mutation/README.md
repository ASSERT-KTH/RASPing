## How to generate mutants of Tracr programs

1. Install required packages

```
pip install -r ../../requirements.txt
cd operators && pip install -e . && cd ..
```

2. Setup configuration file

```toml
[cosmic-ray]
module-path = "source/reverse.py"
timeout = 10.0
excluded-modules = []
test-command = "hammett tests/test_reverse.py"

[cosmic-ray.distributor]
name = "local"
```

This file (`tracr.toml`) defines important information such as:
- Where the source code is (`module-path`)
- How to execute the tests verify whether the change introduces a behavioral change (`test-command`)

3. Create a mutation session

```bash
cosmic-ray init tracr.toml tracr.sqlite
```

This command prepares all the mutations that will later be applied to code.

4. Execute a mutation session

```bash
cosmic-ray exec tracr.toml tracr.sqlite
```

5. Dump mutation results into a JSON file

```bash
cosmic-ray dump tracr.sqlite > tracr.jsonl
```

Important fields:
- `[1][test_outcome]` (`killed` if there a behavioral difference is detected by the tests, `survived` otherwise)
- `[1][diff]` (the diff to be applied to the source file)

Example mutant:
```json
[
    {
        "job_id":"bf5610ea45754f84b2d29b3e6fef2851",
        "mutations":[
            {
                "module_path":"source/reverse.py",
                "operator_name":"core/ReplaceBinaryOperator_Sub_Add",
                "occurrence":0,
                "start_pos":[
                    37,
                    20
                ],
                "end_pos":[
                    37,
                    21
                ],
                "operator_args":{
                    
                }
            }
        ]
    },
    {
        "worker_outcome":"normal",
        "output":"\u001b[31mF\u001b[0m\u001b[31m\n\nFailed: tests.test_reverse.TestReverse.test_reverse\nTraceback (most recent call last):\n  File \"/home/andre/Repos/RASPing/.venv/lib/python3.12/site-packages/hammett/impl.py\", line 549, in run_test\n    resolved_function(**resolved_kwargs)\n  File \"/home/andre/Repos/RASPing/experiments/mutation/tests/test_reverse.py\", line 24, in test_reverse\n    assert result[:-1] == data[1][1:]\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^\nAssertionError\n\n\n\u001b[0m\n--- Local variables ---\ndata:\n    (\n        [\n            'BOS',\n            np.str_('e'),\n            np.str_('b'),\n            np.str_('d'),\n            np.str_('d'),\n            np.str_('c'),\n            np.str_('a'),\n            np.str_('a'),\n            np.str_('e'),\n            np.str_('d'),\n            np.str_('d'),\n        ],\n        [\n            'BOS',\n            np.str_('d'),\n            np.str_('d'),\n            np.str_('e'),\n            np.str_('a'),\n            np.str_('a'),\n            np.str_('c'),\n            np.str_('d'),\n            np.str_('d'),\n            np.str_('b'),\n            np.str_('e'),\n        ],\n    )\nresult:\n    [\n        np.str_('d'),\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n    ]\nself:\n    <tests.test_reverse.TestReverse object at 0x72edfb32d490>\n\n--- Assert components ---\nleft:\n    [\n        np.str_('d'),\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n        None,\n    ]\nright:\n    [\n        np.str_('d'),\n        np.str_('d'),\n        np.str_('e'),\n        np.str_('a'),\n        np.str_('a'),\n        np.str_('c'),\n        np.str_('d'),\n        np.str_('d'),\n        np.str_('b'),\n        np.str_('e'),\n    ]\n\n\u001b[31m0 succeeded, 1 failed, 0 skipped\u001b[0m\n",
        "test_outcome":"killed",
        "diff":"--- mutation diff ---\n--- asource/reverse.py\n+++ bsource/reverse.py\n@@ -34,7 +34,7 @@\n   Returns:\n     reverse : SOp that reverses the input sequence.\n   \"\"\"\n-  opp_idx = (length - rasp.indices).named(\"opp_idx\")\n+  opp_idx = (length + rasp.indices).named(\"opp_idx\")\n   opp_idx = (opp_idx - 1).named(\"opp_idx-1\")\n   reverse_selector = rasp.Select(rasp.indices, opp_idx,\n                                  rasp.Comparison.EQ).named(\"reverse_selector\")"
    }
]
```

6. (Optional) Generate an html report to see the results as a webpage

```bash
cr-html tracr.sqlite > report.html
```