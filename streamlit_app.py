import streamlit as st
import numpy as np
from PIL import Image
import os
import re
import subprocess
import sys
import time
from io import StringIO

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available. Using basic image processing.")

try:
    import pytesseract
    # For cloud deployment, tesseract should be in PATH
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except:
        TESSERACT_AVAILABLE = False
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration - Use Streamlit secrets for deployment
ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT", "https://gpt-4o-intern.openai.azure.com/")
MODEL_NAME = st.secrets.get("MODEL_NAME", "gpt-4.1")
DEPLOYMENT = st.secrets.get("DEPLOYMENT", "gpt-4.1")
API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", "BmaiYil8P7o3Dgv0JzIEIA4JYd3AHl7Jh6SzBdjkwXfF4DNxCzC3JQQJ99BGACYeBjFXJ3w3AAABACOGZkhi")
API_VERSION = st.secrets.get("API_VERSION", "2024-12-01-preview")

# Set page config
st.set_page_config(
    page_title="DSA Problem Solver",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
    )

def preprocess_image_basic(image):
    """Basic image preprocessing without OpenCV"""
    # Convert to grayscale
    img_array = np.array(image.convert('L'))
    return img_array

def preprocess_image_advanced(image):
    """Advanced image preprocessing with OpenCV"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    if not TESSERACT_AVAILABLE:
        return "OCR_PLACEHOLDER_TEXT: Tesseract OCR not available on this deployment. Please type the problem text manually."
        
    try:
        # Preprocess the image
        if CV2_AVAILABLE:
            processed_img = preprocess_image_advanced(image)
        else:
            processed_img = preprocess_image_basic(image)
        
        # Configure tesseract
        custom_config = r'--oem 3 --psm 6'
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        return text.strip() if text.strip() else "No text detected in image"
    except Exception as e:
        return f"OCR Error: {str(e)}. Please type the problem manually."

def get_solution_from_gpt(problem_text):
    """Get solution from GPT-4.1"""
    if not OPENAI_AVAILABLE:
        return "OpenAI library not available. Please install: pip install openai"
        
    client = get_openai_client()
    if not client:
        return "Failed to initialize OpenAI client"
    
    prompt = f"""
    Solve this coding problem use the solution use the given function in a full program to implement the required question and test it on 10 hardcoded test cases and show output it passed or not include edge cases too Give this code without comments and shorter variable names to write this faster in exam

    Problem:
    {problem_text}

    Requirements:
    1. Provide a complete Python program
    2. Include 10 hardcoded test cases with edge cases
    3. Use shorter variable names
    4. No comments in the code
    5. Show pass/fail for each test case
    6. Include proper test execution
    7. Make sure the code runs and prints results clearly
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert competitive programming assistant. Provide clean, efficient code solutions that run correctly.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=13107,
            temperature=0.3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=DEPLOYMENT
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

def analyze_test_failures(stdout):
    """Analyze test case failures to provide detailed feedback"""
    if not stdout:
        return "No output to analyze"
    
    lines = stdout.split('\n')
    failed_tests = []
    passed_tests = []
    
    for line in lines:
        if 'Test' in line and ('FAIL' in line or 'PASS' in line):
            if 'FAIL' in line:
                failed_tests.append(line.strip())
            elif 'PASS' in line:
                passed_tests.append(line.strip())
    
    analysis = f"Test Analysis:\n"
    analysis += f"- Passed: {len(passed_tests)} tests\n"
    analysis += f"- Failed: {len(failed_tests)} tests\n"
    
    if failed_tests:
        analysis += "\nFailing test patterns:\n"
        for test in failed_tests:
            analysis += f"  {test}\n"
        
        # Try to identify patterns in failures
        if len(failed_tests) > 1:
            analysis += "\nPattern analysis:\n"
            # Look for common failure patterns
            if any("Expected=-1" in test for test in failed_tests):
                analysis += "- Some tests expect -1 (impossible/invalid cases)\n"
            if any("Output=-1" in test for test in failed_tests):
                analysis += "- Algorithm returning -1 when it shouldn't\n"
                
    return analysis

def extract_code_snippet(code, max_lines=5):
    """Extract a snippet of the main function for history"""
    lines = code.split('\n')
    func_start = -1
    for i, line in enumerate(lines):
        if 'def ' in line and '(' in line:
            func_start = i
            break
    
    if func_start >= 0:
        snippet_lines = lines[func_start:func_start + max_lines]
        return '\n'.join(snippet_lines) + "..."
    
    return code[:200] + "..." if len(code) > 200 else code

def extract_python_code(response_text):
    """Extract Python code from GPT response"""
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    code_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    return response_text

def run_python_code(code):
    """Execute Python code and capture output - Cloud safe version"""
    try:
        # Create a secure temporary file name
        import tempfile
        import uuid
        temp_file = f'temp_solution_{uuid.uuid4().hex[:8]}.py'
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8'
        )
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Execution timed out (30 seconds)", 1
    except Exception as e:
        return "", str(e), 1

def improve_solution(code, error_output, problem_text, iteration_history=None):
    """Improve solution based on errors and previous iteration history"""
    if not OPENAI_AVAILABLE:
        return "OpenAI library not available for code improvement"
        
    client = get_openai_client()
    if not client:
        return "Failed to initialize OpenAI client"
    
    # Build detailed feedback from iteration history
    history_feedback = ""
    if iteration_history:
        history_feedback = "\n\nPREVIOUS ITERATION ANALYSIS:\n"
        for i, hist in enumerate(iteration_history, 1):
            history_feedback += f"\nIteration {i}:\n"
            history_feedback += f"Code attempted: {hist.get('code_snippet', 'N/A')}\n"
            history_feedback += f"Output: {hist.get('stdout', 'No output')}\n"
            history_feedback += f"Errors: {hist.get('stderr', 'No errors')}\n"
            history_feedback += f"Analysis: {hist.get('analysis', 'No analysis')}\n"
            history_feedback += "-" * 50
    
    prompt = f"""
    You are debugging a coding solution that has failed multiple times. Learn from previous attempts and fix the core issues.

    ORIGINAL PROBLEM:
    {problem_text}

    CURRENT CODE:
    {code}

    CURRENT ERROR OUTPUT:
    {error_output}
    {history_feedback}

    CRITICAL INSTRUCTIONS:
    1. Analyze the pattern of failures across all iterations
    2. Identify the core algorithmic issue, not just syntax problems  
    3. Look at which specific test cases are failing and why
    4. Fix the underlying logic, don't just patch symptoms
    5. Ensure ALL test cases pass (no FAIL outputs)
    6. Use shorter variable names and no comments
    7. Print clear test results showing PASS/FAIL for each test
    8. Include a summary line like "Failed: X" or "All tests passed"

    FOCUS AREAS:
    - If the same test cases keep failing, the algorithm logic is wrong
    - Look at the expected vs actual outputs to understand the pattern
    - Consider edge cases and boundary conditions
    - Make sure you understand what the problem is actually asking for

    Provide a completely corrected solution that addresses the root cause:
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert debugging assistant with access to iteration history. Use the failure patterns to identify and fix root causes, not symptoms.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=13107,
            temperature=0.1,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=DEPLOYMENT
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error during improvement: {str(e)}"

def main():
    st.title("🧠 DSA Problem Solver with OCR")
    st.markdown("Upload images of DSA problems and get automated solutions with test cases!")
    
    # Deployment notice
    st.info("🌐 **Live Demo** - Running on Streamlit Cloud with Azure OpenAI GPT-4.1")
    
    # System status
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if TESSERACT_AVAILABLE:
                st.success("✅ Tesseract OCR")
            else:
                st.error("❌ Tesseract OCR")
        with col2:
            if CV2_AVAILABLE:
                st.success("✅ OpenCV")
            else:
                st.warning("⚠️ OpenCV (basic mode)")
        with col3:
            if OPENAI_AVAILABLE:
                st.success("✅ OpenAI API")
            else:
                st.error("❌ OpenAI API")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        max_iterations = st.slider("Max Improvement Iterations", 1, 20, 5)
        show_intermediate = st.checkbox("Show Intermediate Results", True)
        
        st.header("Display Options")
        code_theme = st.selectbox(
            "Code Theme",
            ["vs-dark", "github", "monokai", "solarized-light", "solarized-dark"],
            index=0
        )
        show_final_code = st.checkbox("Show Final Code in Large Format", True)
        
        st.header("Manual Input")
        manual_text = st.text_area(
            "Or type problem directly:",
            placeholder="Paste or type your DSA problem here...",
            height=150
        )
        
        if manual_text and st.button("🚀 Solve Manual Input"):
            st.session_state['solve_manual'] = manual_text
    
    # Check if manual solve is requested
    if 'solve_manual' in st.session_state:
        problem_text = st.session_state['solve_manual']
        del st.session_state['solve_manual']
        
        st.subheader("📝 Manual Problem Input")
        st.text_area("Problem Text", problem_text, height=150)
        
        # Process the manual input
        process_problem(problem_text, max_iterations, show_intermediate, code_theme, show_final_code)
        return
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload problem images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images containing DSA problems. The app will extract text using OCR."
    )
    
    if uploaded_files:
        st.subheader("📸 Uploaded Images")
        
        # Display uploaded images
        cols = st.columns(min(len(uploaded_files), 3))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image {i+1}", use_container_width=True)
        
        # Extract text from all images
        if st.button("🔍 Extract Text and Solve", type="primary"):
            all_text = ""
            
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                
                with st.spinner(f"Extracting text from Image {i+1}..."):
                    text = extract_text_from_image(image)
                
                if text and not text.startswith("OCR_PLACEHOLDER"):
                    st.success(f"✅ Text extracted from Image {i+1}")
                    if show_intermediate:
                        with st.expander(f"📝 Text from Image {i+1}"):
                            st.text(text)
                    all_text += f"\n\nImage {i+1}:\n{text}"
                elif text.startswith("OCR_PLACEHOLDER"):
                    st.warning(f"⚠️ {text}")
                else:
                    st.warning(f"⚠️ No text found in Image {i+1}")
            
            if all_text.strip():
                st.subheader("🔗 Combined Problem Text")
                st.text_area("Extracted Text", all_text, height=200)
                process_problem(all_text, max_iterations, show_intermediate, code_theme, show_final_code)
            else:
                st.error("❌ No text could be extracted from any image. Try using manual input instead.")

def process_problem(problem_text, max_iterations, show_intermediate, code_theme="vs-dark", show_final_code=True):
    """Process a problem and generate solution"""
    # Get solution from GPT
    with st.spinner("Getting solution from GPT-4.1..."):
        solution = get_solution_from_gpt(problem_text)
    
    if solution and not solution.startswith("OpenAI library not available") and not solution.startswith("API Error"):
        st.subheader("🤖 Generated Solution")
        if show_intermediate:
            st.text_area("GPT Response", solution, height=300)
        
        # Extract and run code
        code = extract_python_code(solution)
        
        if code:
            st.subheader("💻 Extracted Code")
            st.code(code, language='python')
            
            # Execute code iteratively until it passes
            iteration = 0
            max_iter = max_iterations
            last_stdout = ""
            last_stderr = ""
            last_returncode = 0
            iteration_history = []  # Track history for better feedback
            
            while iteration < max_iter:
                iteration += 1
                
                # Show progress
                progress_percentage = iteration / max_iter
                st.progress(progress_percentage, text=f"Iteration {iteration}/{max_iter}")
                
                st.subheader(f"🚀 Execution Attempt {iteration}")
                
                with st.spinner(f"Running code (Attempt {iteration})..."):
                    stdout, stderr, returncode = run_python_code(code)
                    last_stdout, last_stderr, last_returncode = stdout, stderr, returncode
                
                # Analyze current iteration results
                test_analysis = analyze_test_failures(stdout)
                code_snippet = extract_code_snippet(code)
                
                # Add to iteration history
                iteration_history.append({
                    'iteration': iteration,
                    'code_snippet': code_snippet,
                    'stdout': stdout,
                    'stderr': stderr,
                    'returncode': returncode,
                    'analysis': test_analysis
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📤 Output:**")
                    if stdout:
                        st.text_area(f"Output {iteration}", stdout, height=200)
                        
                        # Check if all test cases passed - look for failure indicators
                        failure_indicators = ["FAIL", "failed", "Failed:", "ERROR", "Exception"]
                        has_test_failures = any(fail_word in stdout for fail_word in failure_indicators)
                        
                        # Check for success indicators
                        success_indicators = [
                            "Failed: 0",
                            "All tests passed",
                            "all test cases passed",
                            "Test cases passed: 10"
                        ]
                        has_success = any(indicator in stdout.lower() for indicator in [s.lower() for s in success_indicators])
                        
                        # Only consider it successful if there are success indicators AND no failure indicators
                        if has_success and not has_test_failures:
                            st.success("🎉 All test cases passed!")
                            break
                        elif has_test_failures:
                            st.warning("⚠️ Some test cases are failing")
                        
                        # Show detailed analysis for intermediate results
                        if show_intermediate and test_analysis:
                            with st.expander(f"📊 Test Analysis - Iteration {iteration}"):
                                st.text(test_analysis)
                    else:
                        st.info("No output generated")
                
                with col2:
                    st.markdown("**❌ Errors:**")
                    if stderr:
                        st.text_area(f"Errors {iteration}", stderr, height=200)
                    else:
                        st.success("No errors!")
                
                # Check if we need to improve
                has_errors = bool(stderr) or returncode != 0
                
                # Check for test failures in stdout
                failure_indicators = ["FAIL", "failed", "Failed:", "ERROR", "Exception"]
                has_test_failures = stdout and any(fail_word in stdout for fail_word in failure_indicators)
                
                # Check for success indicators
                success_indicators = [
                    "Failed: 0",
                    "All tests passed", 
                    "all test cases passed",
                    "Test cases passed: 10"
                ]
                has_success = stdout and any(indicator in stdout.lower() for indicator in [s.lower() for s in success_indicators])
                
                # Continue if there are errors OR test failures OR no clear success
                should_continue = has_errors or has_test_failures or not has_success
                
                if should_continue and iteration < max_iter:
                    if has_test_failures:
                        st.warning(f"🔄 Test cases failing, attempting to improve solution... (Attempt {iteration + 1})")
                        st.info(f"📈 Learning from {len(iteration_history)} previous attempts")
                    elif has_errors:
                        st.warning(f"🔄 Code errors detected, attempting to improve solution... (Attempt {iteration + 1})")
                    else:
                        st.warning(f"🔄 No clear success indicators, attempting to improve solution... (Attempt {iteration + 1})")
                    
                    with st.spinner("Improving solution with iteration feedback..."):
                        error_info = f"STDERR: {stderr}\nSTDOUT: {stdout}" if stderr or stdout else "No output generated"
                        improved_solution = improve_solution(code, error_info, problem_text, iteration_history)
                    
                    if improved_solution and not improved_solution.startswith("API Error"):
                        code = extract_python_code(improved_solution)
                        if show_intermediate:
                            st.text_area(f"Improved Code {iteration + 1}", code, height=200)
                            
                            # Show what changed
                            if len(iteration_history) > 0:
                                with st.expander(f"🔄 Changes Made in Iteration {iteration + 1}"):
                                    st.markdown("**Previous approach issues:**")
                                    st.text(iteration_history[-1]['analysis'])
                                    st.markdown("**New approach:**")
                                    st.text("Algorithm logic updated based on failure patterns")
                elif not should_continue:
                    st.success("🎉 Solution completed successfully!")
                    break
                else:
                    st.error("❌ Maximum iterations reached. Manual intervention may be required.")
                    break
            
            # Final summary
            st.subheader("📊 Final Summary")
            
            # Determine final status based on the last iteration
            if iteration <= max_iter:
                # Check the final state using last execution results
                final_has_errors = bool(last_stderr) or (last_returncode != 0)
                final_has_test_failures = last_stdout and any(fail_word in last_stdout for fail_word in ["FAIL", "failed", "Failed:", "ERROR", "Exception"])
                final_has_success = last_stdout and any(indicator in last_stdout.lower() for indicator in ["failed: 0", "all tests passed", "all test cases passed", "test cases passed: 10"])
                
                if final_has_success and not final_has_test_failures and not final_has_errors:
                    final_status = "✅ Success - All test cases passed"
                elif final_has_test_failures:
                    final_status = "⚠️ Partial Success - Some test cases failing"
                elif final_has_errors:
                    final_status = "❌ Failed - Code execution errors"
                else:
                    final_status = "❓ Unknown - Manual review needed"
            else:
                final_status = "❌ Max iterations reached"
                
            st.markdown(f"""
            - **Total Iterations:** {iteration}
            - **Final Status:** {final_status}
            - **Problem Source:** {"Images" if "Image" in problem_text else "Manual Input"}
            - **Learning Applied:** {len(iteration_history)} iterations of feedback used
            - **Success Rate:** {f"{((iteration)/max_iter)*100:.1f}%" if iteration <= max_iter else "100%"}
            """)
            
            # Performance metrics
            if iteration_history:
                st.subheader("⚡ Performance Metrics")
                
                # Calculate test case progression
                passed_counts = []
                failed_counts = []
                
                for hist in iteration_history:
                    stdout = hist.get('stdout', '')
                    if stdout:
                        passed = len([line for line in stdout.split('\n') if 'PASS' in line and 'Test' in line])
                        failed = len([line for line in stdout.split('\n') if 'FAIL' in line and 'Test' in line])
                        passed_counts.append(passed)
                        failed_counts.append(failed)
                    else:
                        passed_counts.append(0)
                        failed_counts.append(0)
                
                if passed_counts and failed_counts:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Initial Test Cases Passed", 
                            passed_counts[0] if passed_counts else 0,
                            help="Test cases that passed in first iteration"
                        )
                    with col2:
                        st.metric(
                            "Final Test Cases Passed", 
                            passed_counts[-1] if passed_counts else 0,
                            delta=passed_counts[-1] - passed_counts[0] if len(passed_counts) > 1 else 0,
                            help="Test cases that passed in final iteration"
                        )
                    with col3:
                        improvement = passed_counts[-1] - passed_counts[0] if len(passed_counts) > 1 else 0
                        st.metric(
                            "Improvement", 
                            f"+{improvement}" if improvement > 0 else improvement,
                            help="Net improvement in test cases passed"
                        )
            
            # Show learning progression if intermediate results are enabled
            if show_intermediate and len(iteration_history) > 1:
                with st.expander("📈 Learning Progression"):
                    for i, hist in enumerate(iteration_history):
                        st.markdown(f"**Iteration {hist['iteration']}:**")
                        if hist['analysis']:
                            st.text(hist['analysis'])
                        st.markdown("---")
            
            # Display final code in large format with syntax highlighting
            if show_final_code and code:
                st.subheader("🎯 Final Solution Code")
                st.markdown("### Complete Working Solution:")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["📋 Formatted Code", "📝 Raw Code"])
                
                with tab1:
                    # Use st.code with language and theme for syntax highlighting
                    st.code(code, language='python', line_numbers=True)
                    
                    # Add copy button functionality
                    st.markdown("**💾 Copy this code for your exam:**")
                    st.text_area(
                        "Select all and copy:",
                        code,
                        height=300,
                        help="Select all text (Ctrl+A) and copy (Ctrl+C)"
                    )
                
                with tab2:
                    # Raw text version for easy copying
                    st.text_area(
                        "Raw code (click and Ctrl+A to select all):",
                        code,
                        height=400,
                        key="raw_code_final"
                    )
                
                # Show code statistics
                lines = code.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Lines", len(lines))
                with col2:
                    st.metric("Code Lines", len(non_empty_lines))
                with col3:
                    st.metric("Characters", len(code))
                with col4:
                    st.metric("Iterations Used", iteration)
                
                # Download button for the code
                st.download_button(
                    label="📥 Download Final Solution",
                    data=code,
                    file_name="dsa_solution.py",
                    mime="text/python",
                    help="Download the final working solution as a Python file"
                )
        
        else:
            st.error("❌ Could not extract Python code from the response")
    else:
        st.error(f"❌ Failed to get solution: {solution}")

if __name__ == "__main__":
    main()
