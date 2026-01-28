class Stack:
  def __init__(self):
    self.stack = []

  def push(self, element):
    self.stack.append(element)

  def pop(self):
    if self.isEmpty():
      return "Stack is empty"
    return self.stack.pop()

  def peek(self):
    if self.isEmpty():
      return "Stack is empty"
    return self.stack[-1]

  def isEmpty(self):
    return len(self.stack) == 0

  def size(self):
    return len(self.stack)

# Create a stack
myStack = Stack()

myStack.push('A')
myStack.push('B')
myStack.push('C')

print("Stack: ", myStack.stack)
print("Pop: ", myStack.pop())
print("Stack after Pop: ", myStack.stack)
print("Peek: ", myStack.peek())
print("isEmpty: ", myStack.isEmpty())
print("Size: ", myStack.size())






# HISTROY_FILE = "history.txt"

# def show_history():
#     file = open(HISTROY_FILE, "r")
#     lines = file.readlines()
#     if len(lines) == 0:
#         print("No history found.")
#     else:
#         for line in reversed(lines):
#             print(line.strip())
#     file.close()


# def clear_history():
#     open(HISTROY_FILE, "w").close()
#     print("History cleared.")


# def save_to_history(equation, result):
#     with open(HISTROY_FILE, "a") as file:
#        file.write(equation +"= " + str(result) + "\n")
#     file.close()

# def calculator(user_input):
#     parts = user_input.split()
#     if len(parts) != 3:
#         print("Invalid input format. Please use: number1 operator number2")
#         return None
#     num1 = float(parts[0])
#     operator = parts[1]
#     num2 = float(parts[2])

#     if operator == "+":
#         result = num1 + num2
#     elif operator == "-":
#         result = num1 - num2
#     elif operator == "*":
#         result = num1 * num2
#     elif operator == "/":
#         if num2 != 0:
#             result = num1 / num2
#         else:
#             print("Error: Division by zero is not allowed.")
#             return None
#     else:
#         print("Invalid operator. Please use +, -, *, or /")
#         return None

#     save_to_history(user_input, result)
#     return result
# def main():
#     print("Welcome to the Calculator!")
#     while True:
#         user_input = input("Enter calculation (or 'history', 'clear', 'exit'): ").strip().lower()
#         if user_input == "exit":
#             print("Exiting the calculator. Goodbye!")
#             break
#         elif user_input == "history":
#             show_history()
#         elif user_input == "clear":
#             clear_history()
#         else:
#             result = calculator(user_input)
#             if result is not None:
#                 print(f"Result: {result}")

# if __name__ == "__main__":
#     main()
    









# import random

# subjects = [
#     "The cat",
#     "A dog",
#     "The teacher",
#     "My friend",
#     "The scientist",
#     "The artist",
#     "The musician",
#     " wood",
#     "The penis",
#     "adult creator"
# ]

# actions = [
#     "jumps over",
#     "runs to",
#     "teaches",
#     "discovers",
#     "paints",
#     "composes",
#     "builds",
#     "explores",
#     "creates",
#     "rides"
# ]

# places_or_things = [
#     "the fence.",
#     "the park.",
#     "the classroom.",
#     "a new theory.",
#     "a beautiful landscape.",
#     "a symphony.",
#     "a skyscraper.",
#     "the universe.",
#     "a masterpiece.",
#     "a viral video."

# ]


# while True:
#     subjects =random.choice(subjects)
#     actions = random.choice(actions)
#     places_or_things = random.choice(places_or_things)

#     headLine = f"BREAKING NEWS: {subjects} {actions} {places_or_things}"
#     print("\n" + headLine + "\n")

#     user_input = input("Generate another headline? (y/n): ").strip().lower()
#     if user_input == 'n':
#         break
    
# print("Thank you for using the headline generator!")




# import tkinter as tk
# from tkinter import filedialog,messagebox


# def new_file():
#     text.delete(1.0, tk.END)

# def open_file():
#     file_path = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Text Files","*.txt")])
#     if file_path:
#         with open(file_path, "r") as file:
#             text.delete(1.0, tk.END)
#             text.insert(tk.End,file.read())


# def save_file():
#     file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files","*.txt")])
#     if file_path:
#         with open(file_path, "w") as file:
#             file.write(text.get(1.0, tk.END))
#             messagebox.showinfo("Save File", "File saved successfully!")

# root = tk.Tk()
# root.title("Text Editor")
# root.geometry("800x600")

# menu = tk.Menu(root)
# root.config(menu=menu)
# file_menu = tk.Menu(menu)
# menu.add_cascade(label="File", menu=file_menu)
# file_menu.add_command(label="New", command=new_file)
# file_menu.add_command(label="Open", command=open_file)
# file_menu.add_command(label="Save", command=save_file)
# text = tk.Text(root, wrap='word', font=("Arial", 12))
# text.pack(expand=1, fill='both')
# root.mainloop()

















#DIGITAL CLOCK USING TKINTER


# import tkinter as tk
# from time import strftime
# from tkinter import Label, Entry, Button

# root = tk.Tk()
# root.title("Digital Clock")


# def time():
#     string = strftime('%H:%M:%S %p \n %D')
#     Label.config(text=string)
#     Label.after(1000,time)


# Label = tk.Label(root, font=("caibri",50,"bold"), background="yellow", foreground="cyan")
# Label.pack(anchor = 'center')

# time()
# root.mainloop()




#AI ASSISTANT USING TKINTER


# import tkinter as tk
# from tkinter import Label, Entry, Button
# import webbrowser

# # define the main window
# root = tk.Tk()
# root.title("YOUR AI ASSISTANT")
# root.geometry("400x200")
# root.configure(bg="steelblue")

# # functions
# def open_youtube():
#     query = entry.get().strip()
#     url = f"https://www.youtube.com/results?search_query={query}"
#     webbrowser.open(url)

# def open_instagram():
#     username = entry.get().replace("@", "").strip()
#     url = f"https://www.instagram.com/{username}/"
#     webbrowser.open(url)

# def open_google():
#     query = entry.get().strip()
#     url = f"https://www.google.com/search?q={query}"
#     webbrowser.open(url)

# # UI
# Label(
#     root,
#     text="Enter your query:",
#     bg="black",
#     fg="white",
#     font=("Arial", 14)
# ).pack(pady=10)

# entry = Entry(root, width=30, font=("Arial", 14))
# entry.pack(pady=5)

# Button(
#     root,
#     text="Search YouTube",
#     command=open_youtube,
#     bg="red",
#     fg="white",
#     font=("Arial", 12)
# ).pack(pady=5)

# Button(
#     root,
#     text="Search Instagram",
#     command=open_instagram,
#     bg="purple",
#     fg="white",
#     font=("Arial", 12)
# ).pack(pady=5)

# Button(
#     root,
#     text="Search Google",
#     command=open_google,
#     bg="green",
#     fg="white",
#     font=("Arial", 12)
# ).pack(pady=5)

# root.mainloop()
