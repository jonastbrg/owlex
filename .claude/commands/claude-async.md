---
description: Run Claude CLI in background with terminal notification when done
---

You are helping the user run Claude CLI asynchronously in the background.

**Usage examples**:
- `/claude-async What are the security risks in this code?`
- `/claude-async Review this API design for best practices`
- `/claude-async Analyze the performance of this function`

**Instructions**:

1. **Take the user's question/request**
2. **Clean old output files** to avoid cluttering /tmp/:
   - Delete files older than 1 hour only (simpler, more compatible)
   ```bash
   find /tmp -name "claude-output-*.txt" -mmin +60 -delete 2>/dev/null
   ```
3. **Create a temporary output file** to store results:
   - Use `/tmp/claude-output-<timestamp>.txt`

4. **Launch Claude in background** using the Bash tool with `run_in_background: true`:
   ```bash
   find /tmp -name "claude-output-*.txt" -mmin +60 -delete 2>/dev/null
   timestamp=$(date +%s) && output_file="/tmp/claude-output-${timestamp}.txt" && \
   (claude exec --skip-git-repo-check --full-auto "QUESTION HERE" > "$output_file" 2>&1 && \
    echo -e "\n========================================" && \
    echo "✅ Claude task completed!" && \
    echo "Results saved to: $output_file" && \
    echo "Use '/claude-async results' to view" && \
    echo "========================================") &
   ```

4. **Return immediately** with a message like:
   ```
   Claude CLI running in background...
   Output will be saved to: /tmp/claude-output-TIMESTAMP.txt

   I'll let you know when it's done!
   Continue working - I'm ready for your next request!
   ```

5. **Auto-notify when complete**: When you see a system reminder that the background bash task has output available, proactively:
   - Check the bash output using BashOutput tool
   - Read the results file
   - Tell the user "Claude task completed!" and show them the results
   - Do this WITHOUT waiting for the user to ask

6. **For manual checking**: If user says "results" or "check", read the most recent output file

**Special handling**:
- If user wants to review specific files, add them to the Claude command
- Append "Be brief unless this requires detailed analysis" by default
- Use proper shell quoting for the question text

**Note**: This runs truly asynchronously - you can continue chatting with Claude while the CLI works in the background!
