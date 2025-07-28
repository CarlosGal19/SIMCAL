FROM node:lts-jod

WORKDIR /home/API_SENSORS

COPY package.json .

RUN npm install

COPY . .

CMD ["npm", "start"]
