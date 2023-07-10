def reverse_string(input_string: str) -> str:
    return input_string[::-1]


def is_prime(n: int) -> bool:
    return all([i%n for i in range(2, n)])


def unique_characters_count(input_string: str) -> int:
    return len(set(input_string))


print(*map(lambda i:'Fizz'*(0==i%3)+'Buzz'*(0==i%5)or i,range(1,101)),sep='\n')


