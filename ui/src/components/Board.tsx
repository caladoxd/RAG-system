import { useState, useEffect, type ChangeEvent } from 'react'

export default function Board() {
    // 10 rows of 5 animal names each, in the format: { name: <animal_name>, selected: false }
    const animalNames: string[][] = [
        ["Lion", "Tiger", "Bear", "Wolf", "Fox"],
        ["Elephant", "Giraffe", "Zebra", "Leopard", "Cheetah"],
        ["Kangaroo", "Koala", "Panda", "Gorilla", "Chimpanzee"],
        ["Hippo", "Rhino", "Crocodile", "Alligator", "Otter"],
        ["Rabbit", "Squirrel", "Mouse", "Rat", "Hamster"],
        ["Horse", "Donkey", "Camel", "Alpaca", "Llama"],
        ["Eagle", "Hawk", "Falcon", "Owl", "Vulture"],
        ["Penguin", "Seagull", "Duck", "Swan", "Goose"],
        ["Shark", "Dolphin", "Whale", "Seal", "Walrus"],
        ["Frog", "Toad", "Turtle", "Lizard", "Snake"]
    ];
    const [timer, setTimer] = useState(0)
    const [startTime, setStartTime] = useState(0)
    const [selectedAnimals, setSelectedAnimals] = useState(animalNames.map(row => row.map(() => false)));
    const [text, setText] = useState("");

    useEffect(() => {
        if(startTime != 0){
            setTimeout(
                () => setTimer(Math.round((Date.now() - startTime)/1000))
            ,1000)
        }
    }, [timer, startTime])

    const checkName = (name: string) => {
        const current: boolean[][] = selectedAnimals;
        for (const i in animalNames) {
            const row = animalNames[i];
            const pos = row.findIndex(animal => animal.toLowerCase() === name.toLowerCase());
            console.log(name, pos);
            if (pos !== -1 && !current[i][pos]) {
                current[i][pos] = true;
                console.log(current);
                setSelectedAnimals(current);
                return true;
            }
        }
        return false;
    }

    const start = async () => {
        setStartTime(Date.now())
    }

    const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
        setText(e.target.value);
        if(checkName(e.target.value)) {
            setText("");
        }
    }
    return (
        <div className="flex flex-col content-center flex-wrap justify-center">
            <input className="m-5 border-gray-300 border-2 rounded-md w-100 " type="text" onChange={handleChange} value={text} />
            <div className="flex flex-row items-center gap-5 my-1">
                <div>
                    Score: {selectedAnimals.reduce((acc, row) => acc + row.filter(e => e).length, 0)} / {animalNames.length * animalNames[0].length}
                </div>
                <div>
                    Time: {timer === 0 ? "0" : timer}
                </div>
                <button onClick={start}>Start</button>
            </div>
            {
                animalNames.map((row, i) => (
                    <div className="flex flex-row items-center gap-5 my-1">
                        {
                            row.map((animal, j) => (
                                <div className="w-40 bg-yellow-100 py-1">
                                    {selectedAnimals[i][j] ? animal : "_"}
                                </div>
                            ))
                        }
                    </div>
                ))
            }
        </div>
    )
}