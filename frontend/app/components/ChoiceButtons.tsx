import { useEffect, useState } from "react"

const ZUNDA_DEFAULT_IMAGE="./zundamon_.png"
const ZUNDA_SELECTED_IMAGE="./zundamon_selected.png"

export function ChoiceButtons() {
    const [currentId, setCurrentId] = useState<number>(1);
    const [id1Selected, setid1Selected] = useState<boolean>(false);
    const [id2Selected, setid2Selected] = useState<boolean>(false);
    const [id3Selected, setid3Selected] = useState<boolean>(false);

    return (
        <div id="button-container">
            <img
                id={currentId === 1 ? "selected-button-icon" : "button-icon"}
                onMouseEnter={() => setid1Selected(true)}
                onMouseLeave={() => setid1Selected(false)}
                onClick={() => {setCurrentId(1);console.log("ID1 Selected!!")}}
                src={id1Selected ? ZUNDA_SELECTED_IMAGE : ZUNDA_DEFAULT_IMAGE}
                alt="id 1"
            />
            <img
                id={currentId === 2 ? "selected-button-icon" : "button-icon"}
                onMouseEnter={() => setid2Selected(true)}
                onMouseLeave={() => setid2Selected(false)}
                onClick={() => {setCurrentId(2);console.log("ID2 Selected!!")}}
                src={id2Selected ? ZUNDA_SELECTED_IMAGE : ZUNDA_DEFAULT_IMAGE}
                alt="id 2"
            />
            <img
                id={currentId === 3 ? "selected-button-icon" : "button-icon"}
                onMouseEnter={() => setid3Selected(true)}
                onMouseLeave={() => setid3Selected(false)}
                onClick={() => {setCurrentId(3);console.log("ID3 Selected!!")}}
                src={id3Selected ? ZUNDA_SELECTED_IMAGE : ZUNDA_DEFAULT_IMAGE}
                alt="id 3"
            />
        </div>
    )
};
