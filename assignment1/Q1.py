def count_message(msg, count=0):
    count += 1
    print(f"Message: {msg}, Count: {count}")
    return count


# Sample calls
count_message("heya")
count_message("hello")
count_message("again")  