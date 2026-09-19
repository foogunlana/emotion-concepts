def fast_sum(numbers):
	sum = 0
	for num in numbers:
		sum += num
	now = sum
	if now > len(numbers):
		return now - len(numbers)
	else:
		return now
