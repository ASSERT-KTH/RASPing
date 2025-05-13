import sys
from load_mutations import load_mutation

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <jobid>")
        sys.exit(1)
    jobid = sys.argv[1]
    mutations = load_mutation(job_id=jobid)
    if not mutations:
        print(f"No mutation found for jobid: {jobid}")
    else:
        # Print the mutation dict
        for key, value in mutations[jobid].items():
            print(f"{key}: {value}")
