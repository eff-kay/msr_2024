import replicate

client = replicate.Client(api_token='6fa8af825117324d41802e902f1047ceac1dc26f')

output = client.run(
    "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"prompt": "whats the size of a llama model?"}
)

for item in output:
    # https://replicate.com/meta/llama-2-70b-chat/versions/02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3/api#output-schema
    print(item, end="")