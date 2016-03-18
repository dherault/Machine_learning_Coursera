'use strict';

const isNumeric = require('./isNumeric');

module.exports = function createIdentityMatrixData(n) {
  if (!isNumeric(n)) throw new Error('createIdentityMatrixData: given size is not a number');
  
  const size = Math.round(Math.abs(n));
  const emptyRow = [];
  const data = [];
  
  for (let i = 0; i < size; i++) {
    emptyRow.push(0);
  }
  for (let i = 0; i < size; i++) {
    data.push(emptyRow.slice());
    data[i][i] = 1;
  }
  
  return data;
};
