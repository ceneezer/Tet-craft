/*
 * map.js
 * To create a digital root table given base and size (to be squared) for "Lost In The Digital Roots" a game by ceneezer
 * Written by Deepseek V3 AI Feb 10th 2025
 * Inspired, co-authored and modified by ceneezer (see bellow comments)
 * Distributed under Creative Commons Licence, modify as you need, and please give public credit to Deepseek and ceneezer.
*/

function generateCity(size, base) {
    // Generate symbols with zero-padding based on the highest digit
    const maxDigitValue = base - 1;
    const digitLength = maxDigitValue.toString().length;
    const symbols = Array.from({length: base}, (_, i) =>
        i.toString().padStart(digitLength, '0')
    );

    // Calculate coordinate range
    start = 1-base;
    end = 0;
    if (size > base-1) {
      start = -maxDigitValue;
      end = size;
    }

    // Generate city grid
    const city = [];
    for (let i = start; i <= end; i++) {
        const row = [];
        for (let j = start; j <= end; j++) {
            if (i === 0 || j === 0) {
                row.push(symbols[0]);
            } else {
                const product = i * j;
                const absProduct = Math.abs(product);
                let modulus = absProduct % (base - 1);

                if (modulus === 0 && product !== 0) {
                    modulus = base - 1;
                }
                row.push(symbols[modulus]);
            }
        }
        city.push(row);
    }

    // Console output for verification
   out=`City for base ${base}:`;
   city.forEach(row => out+="\n"+row.join(' '));
//   console.log(out);
//if (DEBUG===true) {
   console.log(`City for base ${base}:`);
   city.forEach(row => console.log(row.join(' ')));
//}

    return {
        array: city,
        base: base,
        startCoord: start,
        endCoord: end
    };
}

// Test cases
//console.log("--- Test Cases ---");
//generateCity(2+3, 2);
//generateCity(3+3, 3);
//generateCity(4+3, 4);
//generateCity(10+3, 10);
