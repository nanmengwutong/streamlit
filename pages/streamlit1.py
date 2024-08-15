import streamlit as st
def main():
    st.title("Simple To-Do List App")
    # Initialize our to-do list
    if 'todos' not in st.session_state:
        st.session_state.todos = []
    # Input for new to-do item
    new_todo = st.text_input("Add a new to-do item:")
    if st.button("Add") and new_todo:
        st.session_state.todos.append(new_todo)
        st.success(f"Added: {new_todo}")
    # Display the to-do list
    st.subheader("Your To-Do List:")
    for i, todo in enumerate(st.session_state.todos, 1):
        st.write(f"{i}. {todo}")
    # Clear all to-dos
    if st.button("Clear All"):
        st.session_state.todos = []
        st.success("All items cleared!")
if __name__ == "__main__":
    main()
